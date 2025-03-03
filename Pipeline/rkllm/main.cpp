

#include <string.h>
#include <unistd.h>
#include <string>
#include "rkllm.h"
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <chrono>
using namespace std::chrono;
#define PROMPT_TEXT_PREFIX "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
#define PROMPT_TEXT_POSTFIX " [/INST] "

using namespace std;

LLMHandle llmHandle = nullptr;

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





int main(int argc, char **argv)
{
    //if (argc != 2)
    //{
    //    printf("Usage: %s [rkllm_model_path]\n", argv[0]);
    //    return -1;
    //}

    string rkllm_model;
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
        else
        {
            rkllm_model = arg;
        }
    }

    if (rkllm_model.empty())
    {
        printf("Usage: %s [rkllm_model_path] [--file input.txt]\n", argv[0]);
        return -1;
    }




    signal(SIGINT, exit_handler);
    //string rkllm_model(argv[1]);
    printf("RKLLM starting, please wait...\n");

    RKLLMParam param = rkllm_createDefaultParam();
    param.modelPath = rkllm_model.c_str();
    param.target_platform = "rk3588";
    param.num_npu_core = 3;
    param.top_k = 50;
    param.top_p = 0.9;
    param.temperature = 0.8;
    param.repeat_penalty = 1.2;
 //  param.max_new_tokens = 256;
//  param.max_context_len = 512;
 /*   减小加快推理*/
    param.max_new_tokens = 512;
    param.max_context_len = 80;
    std::ofstream("/tmp/start");
    auto start_time = high_resolution_clock::now();
    rkllm_init(&llmHandle, param, callback);
    printf("RKLLM init success!\n");
    std::ofstream("/tmp/stop");
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



    if (file_mode)
    {
        string input_text = read_file(input_file);

        
        string text = PROMPT_TEXT_PREFIX + input_text + PROMPT_TEXT_POSTFIX;
        cout << "Debug: Prompt Sent to Model:\n" << text << endl;
        printf("LLM: ");
        auto start_time = high_resolution_clock::now();
        std::ofstream("/tmp/start");
        rkllm_run(llmHandle, text.c_str(), NULL);
        std::ofstream("/tmp/stop");
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time).count();
        std::cout << "rkllm_run Step Time: " << duration << " ms" << std::endl;

        

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
        printf("LLM: ");
        auto start_time = high_resolution_clock::now();
        std::ofstream("/tmp/start");
        rkllm_run(llmHandle, text.c_str(), NULL);
        std::ofstream("/tmp/stop");
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time).count();
        std::cout << "rkllm_run Step Time: " << duration << " ms" << std::endl;
    }

    rkllm_destroy(llmHandle);

    return 0;
}
