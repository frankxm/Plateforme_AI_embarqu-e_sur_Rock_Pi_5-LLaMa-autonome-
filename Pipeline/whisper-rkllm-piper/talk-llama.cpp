// Talk with AI


#include "common-sdl.h"
#include "common.h"
#include "whisper.h"
#include "llama.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <regex>
#include <string>
#include <thread>
#include <vector>
#include <regex>
#include <sstream>

#include <cstring>
#include <string.h>
#include <unistd.h>
#include <string>
#include "/home/rock/rkllm/include/rkllm.h"
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <chrono>

#include <cstdlib>
using namespace std::chrono;

// #define PROMPT_TEXT_PREFIX "<|start|><|system|> You are a helpful assistant. <|end|> <|start|><|user|>"
// #define PROMPT_TEXT_POSTFIX "<|end|><|start|><|assistant|>"
#define PROMPT_TEXT_PREFIX "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
#define PROMPT_TEXT_POSTFIX " [/INST] "
// #define PROMPT_TEXT_POSTFIX_ASSITANT "<|end|><|start|><|user|>"
using namespace std;

LLMHandle llmHandle = nullptr;
std::string model_response; // 存储模型推理的响应

// 全局管道变量
FILE* piperPipe = nullptr; 



// 使用正则表达式分割文本
std::vector<std::string> splitByPunctuation(const std::string& text) {
    std::vector<std::string> sentences;
    std::regex re(R"([,.;!？。，:])"); // 匹配逗号、句号、感叹号、分号等标点符号
    std::sregex_token_iterator iter(text.begin(), text.end(), re, -1);
    std::sregex_token_iterator end;

    for (; iter != end; ++iter) {
        std::string trimmed = iter->str();
        // 去掉首尾多余的空格
        trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);
        trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
        if (!trimmed.empty()) {
            sentences.push_back(trimmed);
        }
    }

    return sentences;
}

void startPiper() {
    const std::string piperCommand = "~/piper/piper --model ~/piper/GB_female_south/en_GB-southern_english_female-low.onnx --output-raw --debug | aplay --buffer-time=2000000 -r 15000 -f S16_LE -t raw -";
    piperPipe = popen(piperCommand.c_str(), "w");
    if (!piperPipe) {
        std::cerr << "Error: Failed to start Piper." << std::endl;
    } else {
        std::cout << "Piper started successfully." << std::endl;
    }
    printf("start piper thread ID: %ld\n", std::this_thread::get_id());
}

void stopPiper() {
    if (piperPipe) {
        int returnCode = pclose(piperPipe);
        piperPipe = nullptr;
        if (returnCode == -1) {
            std::cerr << "Error: Failed to close Piper." << std::endl;
        } else {
            std::cout << "Piper closed successfully." << std::endl;
        }
    }
}

void streamOutputWithPiper(const std::string& model_response) {
    if (!piperPipe) {
        std::cerr << "Warning: Piper is not running. Starting Piper now..." << std::endl;
        startPiper();
        if (!piperPipe) {
            std::cerr << "Error: Failed to start Piper. Cannot process model response." << std::endl;
            return;
        }
    }
    printf("piper thread ID: %ld\n", std::this_thread::get_id());
    std::vector<std::string> sentences = splitByPunctuation(model_response);

    for (const auto& sentence : sentences) {
        std::cout << "Sending to Piper: " << sentence << std::endl;
        fwrite(sentence.c_str(), sizeof(char), sentence.size(), piperPipe);
        fwrite("\n", sizeof(char), 1, piperPipe);
        fflush(piperPipe);
    }
}


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
    // 模型推理为单线程  callback 是由 rkllm_run 同步触发的（即 rkllm_run 会阻塞直到回调完成
    // printf("Callback thread ID the callback: %ld\n", std::this_thread::get_id());
    if (state == LLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (state == LLM_RUN_ERROR)
    {
        printf("\\LLM run error\n");
    }
    // 推理过程中不断把每个结果输出，每次回调只返回一个token
    else
    {
        // 将生成的响应追加到 model_response
        model_response += text;
        printf("%s", text);
    }
}





// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t voice_ms   = 10000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;
    int32_t n_gpu_layers = 999;

    float vad_thold  = 0.5f;
    float freq_thold = 100.0f;

    bool translate      = false;
    bool print_special  = false;
    bool print_energy   = false;
    bool no_timestamps  = true;
    bool verbose_prompt = false;
    bool use_gpu        = false;
    bool flash_attn     = false;

    std::string person      = "Georgi";
    std::string bot_name    = "LLaMA";
    std::string wake_cmd    = "";
    std::string heard_ok    = "";
    std::string language    = "en";
    std::string model_wsp   = "models/ggml-base.en.bin";
    std::string model_llama = "~/rkllm/llama2-chat-7b-hf-002.rkllm";
    std::string speak       = "./examples/talk-llama/speak";
    std::string speak_file  = "./examples/talk-llama/to_speak.txt";
    std::string prompt      = "";
    std::string fname_out;
    std::string path_session = "";       // path to file for saving/loading model eval state
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"   || arg == "--threads")        { params.n_threads      = std::stoi(argv[++i]); }
        else if (arg == "-vms" || arg == "--voice-ms")       { params.voice_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"   || arg == "--capture")        { params.capture_id     = std::stoi(argv[++i]); }
        else if (arg == "-mt"  || arg == "--max-tokens")     { params.max_tokens     = std::stoi(argv[++i]); }
        else if (arg == "-ac"  || arg == "--audio-ctx")      { params.audio_ctx      = std::stoi(argv[++i]); }
        else if (arg == "-ngl" || arg == "--n-gpu-layers")   { params.n_gpu_layers   = std::stoi(argv[++i]); }
        else if (arg == "-vth" || arg == "--vad-thold")      { params.vad_thold      = std::stof(argv[++i]); }
        else if (arg == "-fth" || arg == "--freq-thold")     { params.freq_thold     = std::stof(argv[++i]); }
        else if (arg == "-tr"  || arg == "--translate")      { params.translate      = true; }
        else if (arg == "-ps"  || arg == "--print-special")  { params.print_special  = true; }
        else if (arg == "-pe"  || arg == "--print-energy")   { params.print_energy   = true; }
        else if (arg == "-vp"  || arg == "--verbose-prompt") { params.verbose_prompt = true; }
        else if (arg == "-ng"  || arg == "--no-gpu")         { params.use_gpu        = false; }
        else if (arg == "-fa"  || arg == "--flash-attn")     { params.flash_attn     = true; }
        else if (arg == "-p"   || arg == "--person")         { params.person         = argv[++i]; }
        else if (arg == "-bn"   || arg == "--bot-name")      { params.bot_name       = argv[++i]; }
        else if (arg == "--session")                         { params.path_session   = argv[++i]; }
        else if (arg == "-w"   || arg == "--wake-command")   { params.wake_cmd       = argv[++i]; }
        else if (arg == "-ho"  || arg == "--heard-ok")       { params.heard_ok       = argv[++i]; }
        else if (arg == "-l"   || arg == "--language")       { params.language       = argv[++i]; }
        else if (arg == "-mw"  || arg == "--model-whisper")  { params.model_wsp      = argv[++i]; }
        else if (arg == "-ml"  || arg == "--model-llama")    { params.model_llama    = argv[++i]; }
        else if (arg == "-s"   || arg == "--speak")          { params.speak          = argv[++i]; }
        else if (arg == "-sf"  || arg == "--speak-file")     { params.speak_file     = argv[++i]; }
        else if (arg == "--prompt-file")                     {
            std::ifstream file(argv[++i]);
            std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), back_inserter(params.prompt));
            if (params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
        else if (arg == "-f"   || arg == "--file")          { params.fname_out     = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,       --help           [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,     --threads N      [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -vms N,   --voice-ms N     [%-7d] voice duration in milliseconds\n",              params.voice_ms);
    fprintf(stderr, "  -c ID,    --capture ID     [%-7d] capture device ID\n",                           params.capture_id);
    fprintf(stderr, "  -mt N,    --max-tokens N   [%-7d] maximum number of tokens per audio chunk\n",    params.max_tokens);
    fprintf(stderr, "  -ac N,    --audio-ctx N    [%-7d] audio context size (0 - all)\n",                params.audio_ctx);
    fprintf(stderr, "  -ngl N,   --n-gpu-layers N [%-7d] number of layers to store in VRAM\n",           params.n_gpu_layers);
    fprintf(stderr, "  -vth N,   --vad-thold N    [%-7.2f] voice activity detection threshold\n",        params.vad_thold);
    fprintf(stderr, "  -fth N,   --freq-thold N   [%-7.2f] high-pass frequency cutoff\n",                params.freq_thold);
    fprintf(stderr, "  -tr,      --translate      [%-7s] translate from source language to english\n",   params.translate ? "true" : "false");
    fprintf(stderr, "  -ps,      --print-special  [%-7s] print special tokens\n",                        params.print_special ? "true" : "false");
    fprintf(stderr, "  -pe,      --print-energy   [%-7s] print sound energy (for debugging)\n",          params.print_energy ? "true" : "false");
    fprintf(stderr, "  -vp,      --verbose-prompt [%-7s] print prompt at start\n",                       params.verbose_prompt ? "true" : "false");
    fprintf(stderr, "  -ng,      --no-gpu         [%-7s] disable GPU\n",                                 params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,      --flash-attn     [%-7s] flash attention\n",                             params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -p NAME,  --person NAME    [%-7s] person name (for prompt selection)\n",          params.person.c_str());
    fprintf(stderr, "  -bn NAME, --bot-name NAME  [%-7s] bot name (to display)\n",                       params.bot_name.c_str());
    fprintf(stderr, "  -w TEXT,  --wake-command T [%-7s] wake-up command to listen for\n",               params.wake_cmd.c_str());
    fprintf(stderr, "  -ho TEXT, --heard-ok TEXT  [%-7s] said by TTS before generating reply\n",         params.heard_ok.c_str());
    fprintf(stderr, "  -l LANG,  --language LANG  [%-7s] spoken language\n",                             params.language.c_str());
    fprintf(stderr, "  -mw FILE, --model-whisper  [%-7s] whisper model file\n",                          params.model_wsp.c_str());
    fprintf(stderr, "  -ml FILE, --model-llama    [%-7s] llama model file\n",                            params.model_llama.c_str());
    fprintf(stderr, "  -s FILE,  --speak TEXT     [%-7s] command for TTS\n",                             params.speak.c_str());
    fprintf(stderr, "  -sf FILE, --speak-file     [%-7s] file to pass to TTS\n",                         params.speak_file.c_str());
    fprintf(stderr, "  --prompt-file FNAME        [%-7s] file with custom prompt to start dialog\n",     "");
    fprintf(stderr, "  --session FNAME                   file to cache model state in (may be large!) (default: none)\n");
    fprintf(stderr, "  -f FNAME, --file FNAME     [%-7s] text output file name\n",                       params.fname_out.c_str());
    fprintf(stderr, "\n");
}



static std::string transcribe(
        whisper_context * ctx,
        const whisper_params & params,
        const std::vector<float> & pcmf32,
        const std::string & prompt_text) {
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);


    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.no_context       = false;
    wparams.single_segment   = true;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;
    wparams.audio_ctx        = params.audio_ctx;

    // 如果提供提示文本，将其作为上下文提示
    std::vector<whisper_token> prompt_tokens;
    if (!prompt_text.empty()) {
        prompt_tokens.resize(1024);
        prompt_tokens.resize(whisper_tokenize(ctx, prompt_text.c_str(), prompt_tokens.data(), prompt_tokens.size()));
        wparams.prompt_tokens   = prompt_tokens.data();
        wparams.prompt_n_tokens = prompt_tokens.size();
    }

    // 转录音频
    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
         fprintf(stderr, "error transcription\n");
        return ""; // 转录失败
    }

    std::string result;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        result += whisper_full_get_segment_text(ctx, i);
    }
    
    std::cout << "Detecting audio, audio size: " << pcmf32.size() << "\n";

    return result; // 返回转录文本
}




int main(int argc, char ** argv) {
    int ret=system("sudo bash fix_freq_rk3588.sh");
    if(ret==-1){
        std:cerr<<"Error: Failed to execute fix_freq_rk3588.sh"<<std::endl;
    }
    else{
        std::cout<<"fix_freq_rk3588.sh executed sucessfully"<<std::endl;
    }
    // 判断模型推理是否为多线程 单
    printf("Main thread ID: %ld\n", std::this_thread::get_id());
    signal(SIGINT, exit_handler);
    string rkllm_model("/home/rock/rkllm/llama2-chat-7b-hf-002.rkllm" );
    printf("RKLLM starting, please wait...\n");

    RKLLMParam param = rkllm_createDefaultParam();
    param.modelPath = rkllm_model.c_str();
    param.target_platform = "rk3588";
    param.num_npu_core = 3;
    param.top_k = 50;
    param.top_p=0.8;
    param.temperature=0.8;
    param.repeat_penalty=1.2;
 /*   减小加快推理*/
    param.max_new_tokens = 256;
    param.max_context_len = 128;

    auto start_time = high_resolution_clock::now();
    rkllm_init(&llmHandle, param, callback);
    printf("RKLLM init success!\n");
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time).count();
    std::cout << "rkllm_init Step Time: " << duration << " ms" << std::endl;

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



    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context * ctx_wsp = whisper_init_from_file_with_params(params.model_wsp.c_str(), cparams);
    if (!ctx_wsp) {
        fprintf(stderr, "No whisper.cpp model specified. Please provide using -mw <modelfile>\n");
        return 1;
    }

    audio_async audio(30*1000);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return 1;
    }

    audio.resume();

    bool is_running = true;
    std::vector<float> pcmf32_cur;

    printf("%s : start speaking in the microphone\n", __func__);
    int64_t t_ms = 0;
    float prob0 = 0.0f;
    // std::string alltext = PROMPT_TEXT_PREFIX;
    // int transcription_count = 0; // 记录转录次数
    // const int reset_threshold =5; // 达到这个次数后重置 alltext

// print some info about the processing
    {
        fprintf(stderr, "\n");

        if (!whisper_is_multilingual(ctx_wsp)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing, %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "\n");
    }
    
    while (is_running) {


        is_running = sdl_poll_events();
        if (!is_running) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        audio.pause();
        audio.resume();
        audio.get(2000, pcmf32_cur);
        printf("Audio buffer size: %lu\n", pcmf32_cur.size());

        bool vad_detected = vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 2000, params.vad_thold, params.freq_thold, params.print_energy);
        printf("VAD detection result: %d\n", vad_detected);
        //进行语音转录前，调用 vad_simple 检测语音活动
        if (::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 1250, params.vad_thold, params.freq_thold, params.print_energy)) {
            printf("Speech detected! Processing transcription...\n");
            audio.get(params.voice_ms, pcmf32_cur);
            printf("Audio buffer size: %lu\n", pcmf32_cur.size());

            std::string transcribed_text = transcribe(ctx_wsp, params, pcmf32_cur, "");
            if (!transcribed_text.empty()&&transcribed_text.length()>0) {
               
                printf("\nUser (Transcribed): %s\n", transcribed_text.c_str());
                fflush(stdout);

                
                std::string text_cur = PROMPT_TEXT_PREFIX+transcribed_text+PROMPT_TEXT_POSTFIX;
                cout << "Debug: Prompt Sent to Model:\n" << text_cur << endl;
                printf("Length the prompt: %zu\n", text_cur.length());
                fflush(stdout);
                // 调用 RKLLM 模型生成回复
                printf("LLM: ");
                auto start_time = high_resolution_clock::now();
                // 通过初始化时的callback回调函数进行实时推理
                rkllm_run(llmHandle, text_cur.c_str(), NULL);
                auto end_time = high_resolution_clock::now();
                auto duration = duration_cast<milliseconds>(end_time - start_time).count();
                std::cout << "rkllm_run Step Time: \n" << duration << " ms" << std::endl;

                // 使用模型响应更新 alltext
                if (!model_response.empty()) {
                    printf("\n model_response: %s saved for piper \n", model_response.c_str());
                    fflush(stdout);
                    printf("rkllm thread ID: %ld\n", std::this_thread::get_id());
                    streamOutputWithPiper(model_response);
                    stopPiper(); // 关闭 Piper
                    // alltext += model_response+PROMPT_TEXT_POSTFIX_ASSITANT;
                    model_response.clear(); // 清空响应，准备下一轮
                }

                std::cout << "Finish inference, please speak: " << "\n";
        }


            audio.clear();
        }
    }
    
    rkllm_destroy(llmHandle);
    audio.pause();

    whisper_print_timings(ctx_wsp);
    whisper_free(ctx_wsp);

    return 0;
}

