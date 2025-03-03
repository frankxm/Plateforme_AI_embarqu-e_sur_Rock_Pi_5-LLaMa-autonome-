#include "arg.h"
#include "common.h"
#include "console.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include "llama-vocab.h"
#include <unordered_map>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif






#include <string.h>
#include "rkllm.h"
#include <csignal>
#include <chrono>
using namespace std::chrono;

#define PROMPT_TEXT_PREFIX "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
#define PROMPT_TEXT_POSTFIX " [/INST] "



using namespace std;

LLMHandle llmHandle = nullptr;
std::string model_response; 

void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        cout << "Catched exit signal. Exiting..." << endl;
        LLMHandle _tmp = llmHandle;
        llmHandle = nullptr;
        rkllm_destroy(_tmp);
        exit(signal);
    }
}

void callback(const char *text, void *userdata, LLMCallState state)
{
    if (state == LLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (state == LLM_RUN_ERROR)
    {
        printf("\\LLM run error\n");
    }
    else
    {
        model_response += text;
        printf("%s", text);
    }
}

string read_file(const string& filename)
{
    ifstream file(filename);
    if (!file)
    {
        cerr << "Error: Unable to open file " << filename << endl;
        exit(-1);
    }

    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    return content;
}











static llama_context           ** g_ctx;
static llama_model             ** g_model;
static common_sampler          ** g_smpl;
static common_params            * g_params;
static std::vector<llama_token> * g_input_tokens;
static std::ostringstream       * g_output_ss;
static std::vector<llama_token> * g_output_tokens;
static bool is_interacting  = false;
static bool need_insert_eot = false;

static void print_usage(int argc, char ** argv) {
    (void) argc;

    LOG("\nexample usage:\n");
    LOG("\n  text generation:     %s -m your_model.gguf -p \"I believe the meaning of life is\" -n 128\n", argv[0]);
    LOG("\n  chat (conversation): %s -m your_model.gguf -p \"You are a helpful assistant\" -cnv\n", argv[0]);
    LOG("\n");
}

static bool file_exists(const std::string & path) {
    std::ifstream f(path.c_str());
    return f.good();
}

static bool file_is_empty(const std::string & path) {
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_params->interactive) {
            is_interacting  = true;
            need_insert_eot = true;
        } else {
            console::cleanup();
            LOG("\n");
            common_perf_print(*g_ctx, *g_smpl);

            // make sure all logs are flushed
            LOG("Interrupted by user\n");
            common_log_pause(common_log_main());

            _exit(130);
        }
    }
}
#endif

static std::string chat_add_and_format(struct llama_model * model, std::vector<common_chat_msg> & chat_msgs, const std::string & role, const std::string & content) {
    common_chat_msg new_msg{role, content};
    auto formatted = common_chat_format_single(model, g_params->chat_template, chat_msgs, new_msg, role == "user");
    chat_msgs.push_back({role, content});
    LOG_INF("formatted: '%s'\n", formatted.c_str());
    return formatted;
}
void save_vocab_to_file(const llama_vocab *vocab, const std::string &filename) {
    if (!vocab) {
        std::cerr << "Error: vocab is NULL!" << std::endl;
        return;
    }

    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    outfile << "=== LLaMA Vocabulary ===\n";
    outfile << "Total Tokens: " << vocab->n_tokens() << "\n\n";

    for (uint32_t i = 0; i < vocab->n_tokens(); ++i) {
        const char *token_text = vocab->token_get_text(i);
        outfile << "Token[" << i << "]: " << (token_text ? token_text : "(null)") << "\n";
    }

    outfile.close();
    std::cout << "Vocabulary saved to " << filename << std::endl;
}



int main(int argc, char ** argv) {
    string rkllm_model;
    rkllm_model = "/home/rock/rkllm/llama2-chat-7b-hf-002.rkllm";
    string input_file;
    bool file_mode = false;
    // 解析命令行参数
    for (int i = 1; i < argc; i++)
    {
        string arg = argv[i];

        if (arg == "--file")
        {
            if (i + 1 < argc)
            {
                input_file = argv[i + 1];
                file_mode = true;
                i++; // 跳过文件名
            }
            else
            {
                cerr << "Error: Missing file path after --file\n";
                return -1;
            }
        }

    }


    common_params params;
    g_params = &params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MAIN, print_usage)) {
        return 1;
    }

    common_init();

    auto & sparams = params.sampling;

    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    console::init(params.simple_io, params.use_color);
    atexit([]() { console::cleanup(); });

    if (params.logits_all) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.embedding) {
        LOG_ERR("************\n");
        LOG_ERR("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        LOG_ERR("************\n\n");

        return 0;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_INF("%s: llama backend init\n", __func__);

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    common_sampler * smpl = nullptr;

    g_model = &model;
    g_ctx = &ctx;
    g_smpl = &smpl;

    std::vector<common_chat_msg> chat_msgs;


    // load the model and apply lora adapter, if any
    LOG_INF("%s: load the model and apply lora adapter, if any\n", __func__);
    common_init_result llama_init = common_init_from_params(params);
    model = llama_init.model.get();
    ctx = llama_init.context.get();
    if (model == NULL) {
        LOG_ERR("%s: error: unable to load model\n", __func__);
        return 1;
    }
    

    const llama_vocab * vocab = llama_model_get_vocab(model);
    printf("Total number of token :%d\n",vocab->n_tokens());

    LOG_INF("%s: llama threadpool init, n_threads = %d\n", __func__, (int) params.cpuparams.n_threads);

    
    signal(SIGINT, exit_handler);
    printf("RKLLM starting, please wait...\n");

    RKLLMParam param = rkllm_createDefaultParam();
    param.modelPath = rkllm_model.c_str();
    param.target_platform = "rk3588";
    param.num_npu_core = 3;
    param.top_k = 50;
    param.top_p = 0.9;
    param.temperature = 0.8;
    param.repeat_penalty = 1.2;
 /*   减小加快推理*/
    param.max_new_tokens = 512;
    param.max_context_len = 80;
    std::ofstream("/tmp/start");
    auto start_time1 = high_resolution_clock::now();
    rkllm_init(&llmHandle, param, callback);
    printf("RKLLM init success!\n");
    std::ofstream("/tmp/stop");
    auto end_time1 = high_resolution_clock::now();
    auto duration_load = duration_cast<milliseconds>(end_time1 - start_time1).count();
    std::cout << "rkllm_init Step Time: " << duration_load << " ms" << std::endl;

    vector<string> pre_input;
    pre_input.push_back("Welcome to ezrkllm! This is an adaptation of Rockchip's rknn-llm repo (see github.com/airockchip/rknn-llm) for running LLMs on its SoCs' NPUs.\n");
    pre_input.push_back("You are currently running the runtime for ");
    pre_input.push_back(param.target_platform);
    pre_input.push_back("\nTo exit the model, enter either exit or quit\n");
    pre_input.push_back("\nMore information here: https://github.com/Pelochus/ezrknpu");
    pre_input.push_back("\nDetailed information for devs here: https://github.com/Pelochus/ezrknn-llm");

    cout << "\n*************************** Pelochus' ezrkllm runtime *************************\n" << endl;

    for (int i = 0; i < (int)pre_input.size(); i++)
    {
        cout << pre_input[i];
    }

    cout << "\n*******************************************************************************\n" << endl;

    
    printf("num_npu_core = %d, max_context_len=%d, max_new_tokens=%d, top_k = %d, top_p = %.3f, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty= %.3f, mirostat = %.3f, mirostat_tau = %.3d, mirostat_eta = %.3f\n",param.num_npu_core,param.max_context_len,param.max_new_tokens,
    param.top_k,param.top_p,param.temperature,param.repeat_penalty,param.frequency_penalty,param.presence_penalty,param.mirostat,param.mirostat_tau,param.mirostat_eta);



    if (file_mode)
    {
        string input_text = read_file(input_file);


        string text = PROMPT_TEXT_PREFIX + input_text + PROMPT_TEXT_POSTFIX;
        cout << "Debug: Prompt Sent to Model:\n" << text << endl;
        printf("tokenize the prompt\n");
        printf("prompt=%s\n",input_text.c_str());
        auto start_time2 = high_resolution_clock::now();
        std::vector<llama_token> embd_inp2 = common_tokenize(ctx, input_text, true, true);
        auto end_time2 = high_resolution_clock::now();
        auto duration_prompt = duration_cast<milliseconds>(end_time2 - start_time2).count();
        std::cout << "Prompt eval time: " << duration_prompt << " ms" << std::endl;
        printf("tokens: %s\n toknezed result size: %ld\n", string_from(ctx, embd_inp2).c_str(),embd_inp2.size());
        printf("LLM: ");
        auto start_time3 = high_resolution_clock::now();
        std::ofstream("/tmp/start");
        rkllm_run(llmHandle, text.c_str(), NULL);
        std::ofstream("/tmp/stop");
        auto end_time3 = high_resolution_clock::now();
        auto duration_inference = duration_cast<milliseconds>(end_time3 - start_time3).count();
        std::cout << "rkllm_run Step Time: " << duration_inference << " ms" << std::endl;
        if (!model_response.empty()) {
            std::vector<llama_token> embd_inp3 = common_tokenize(ctx, model_response, true, true);
            printf("tokens: %s\n toknezed result size:%ld\n", string_from(ctx, embd_inp3).c_str(),embd_inp3.size());
            model_response.clear();

            printf("load time = %10ld ms\n",duration_load);
            printf("prompt eval time = %10ld ms / %5ld tokens (%8.2f ms per token, %8.2f tokens per second)\n",duration_prompt, embd_inp2.size(),(double)duration_prompt / embd_inp2.size(), (1e3 * (double)embd_inp2.size() / duration_prompt));
            printf("eval time = %10ld ms / %5ld runs   (%8.2f ms per token, %8.2f tokens per second)\n",duration_inference, embd_inp3.size(),(double)duration_inference / embd_inp3.size(), (1e3 * (double)embd_inp3.size() / duration_inference));

        
        }

        
        

        rkllm_destroy(llmHandle);
        return 0; // **退出，不进入交互模式**
    }
    

    while (true)
    {
        std::string input_str;
        printf("\n");
        printf("You: ");
        std::getline(std::cin, input_str);

        if (input_str == "exit" || input_str == "quit")
        {
            cout << "Quitting program..." << endl;
            
            break;
        }

        for (int i = 0; i < (int)pre_input.size(); i++)
        {
            if (input_str == to_string(i))
            {
                input_str = pre_input[i];
                cout << input_str << endl;
            }
        }
        
        string text = PROMPT_TEXT_PREFIX + input_str + PROMPT_TEXT_POSTFIX;
        cout << "Debug: Prompt Sent to Model:\n" << text << endl;
        printf("tokenize the prompt\n");
        printf("prompt=%s\n",input_str.c_str());
        auto start_time2 = high_resolution_clock::now();
        std::vector<llama_token> embd_inp2 = common_tokenize(ctx, input_str, true, true);
        auto end_time2 = high_resolution_clock::now();
        auto duration_prompt = duration_cast<milliseconds>(end_time2 - start_time2).count();
        std::cout << "Prompt eval time: " << duration_prompt << " ms" << std::endl;
        printf("tokens: %s\n toknezed result size: %ld\n", string_from(ctx, embd_inp2).c_str(),embd_inp2.size());
        printf("LLM: ");
        auto start_time3 = high_resolution_clock::now();
        std::ofstream("/tmp/start");
        rkllm_run(llmHandle, text.c_str(), NULL);
        std::ofstream("/tmp/stop");
        auto end_time3 = high_resolution_clock::now();
        auto duration_inference = duration_cast<milliseconds>(end_time3 - start_time3).count();
        std::cout << "rkllm_run Step Time: " << duration_inference << " ms" << std::endl;
        if (!model_response.empty()) {
            std::vector<llama_token> embd_inp3 = common_tokenize(ctx, model_response, true, true);
            printf("tokens: %s\n toknezed result size:%ld\n", string_from(ctx, embd_inp3).c_str(),embd_inp3.size());
            model_response.clear();

            printf("load time = %10ld ms\n",duration_load);
            printf("prompt eval time = %10ld ms / %5ld tokens (%8.2f ms per token, %8.2f tokens per second)\n",duration_prompt, embd_inp2.size(),(double)duration_prompt / embd_inp2.size(), (1e3 * (double)embd_inp2.size() / duration_prompt));
            printf("eval time = %10ld ms / %5ld runs   (%8.2f ms per token, %8.2f tokens per second)\n",duration_inference, embd_inp3.size(),(double)duration_inference / embd_inp3.size(), (1e3 * (double)embd_inp3.size() / duration_inference));

        
        }
    }

    rkllm_destroy(llmHandle);

    return 0;
    
    
}
