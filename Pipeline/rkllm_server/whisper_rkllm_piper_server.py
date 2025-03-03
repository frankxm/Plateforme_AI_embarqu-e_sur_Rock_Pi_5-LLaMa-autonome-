from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import gradio as gr
import torch
import torchaudio
import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import gradio as gr
import argparse
import queue
import subprocess
import uuid
import base64
import re
import concurrent.futures
import multiprocessing
import wave
from pydub import AudioSegment

# 自定义 Gradio UI 布局

css = """
.gradio-container{
background:radial-gradient(#416e8a, #000000);
}
#button{
background:#06354d
}
#stop-btn {
    width: 200px;
    height: 40px;
    padding: 5px;
    font-size: 14px;
    border-radius: 5px;
    background:#06354d;
    display: flex;
    align-items: center; /* 垂直居中 */
    justify-content: center; /* 水平居中 */
    margin: auto; /* 自动外边距让按钮居中 */
}

.markdown {
    color: white;
    font-size: 16px;
}
"""

#PROMPT_TEXT_PREFIX = "<|im_start|>system You are a helpful assistant. <|im_end|> <|im_start|>user"
#PROMPT_TEXT_POSTFIX = "<|im_end|><|im_start|>assistant"
#PROMPT_TEXT_PREFIX = "<|start|><|system|> You are a helpful assistant. <|end|> <|start|><|user|>"
#PROMPT_TEXT_POSTFIX = "<|end|><|start|><|assistant|>"
PROMPT_TEXT_PREFIX ="<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
PROMPT_TEXT_POSTFIX =" [/INST] "
# Set environment variables
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
#os.environ["GRADIO_SERVER_NAME"] = "172.28.190.190"
os.environ["GRADIO_SERVER_PORT"] = "8080"

# Set the dynamic library path
try:
	rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')
	print(f"success loading dynamic library")
except OSError as e:
        print(f"Error loading dynamic library: {e}")
        sys.exit(1)
# Define the structures from the library
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)
class LLMCallState:
    LLM_RUN_NORMAL = 0
    LLM_RUN_FINISH = 1
    LLM_RUN_ERROR = 2






class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("target_platform", ctypes.c_char_p),
        ("num_npu_core", ctypes.c_int32),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
    ]


# 全局变量存callback返回的值
global_text = []
global_state = -1

#  用于中断推理
stop_event = threading.Event()


# 创建全局队列，存储待播放的文本
piper_queue = queue.Queue()

input_empty=False




def stop_RKLLM():
    stop_event.set()  # 设置中断信号

def callback_impl(result, userdata, state):

    # print(f"callback ID: {threading.get_ident()}, name: {threading.current_thread().name}")
    global global_text, global_state
    if state == LLMCallState.LLM_RUN_FINISH:
    	global_state = state
    	print("\n")
    elif state == LLMCallState.LLM_RUN_ERROR:
        global_state = state
        print("LLM run error")
    else:
        global_state = state
    	# 将结果解码为字符串并添加到 global_text 中
        try:
            # c++返回字节类型，转成字符串类型
            global_text.append(result.decode('utf-8'))
            # print("result not decode:",result)
            # print("result decode:",result.decode('utf-8'))
        except Exception as e:
            print(f"Error in callback: {e}")


# Connect the callback function between the Python side and the C++ side

callback_type = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)



# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
    def __init__(self, model_path):

        rkllm_param = RKLLMParam()

        rkllm_param.model_path = bytes(model_path, 'utf-8')
        #额外加的
        rkllm_param.target_platform = b"rk3588"
        rkllm_param.max_context_len =80
        rkllm_param.max_new_tokens = 256

        rkllm_param.top_k = 50
        rkllm_param.top_p = 0.9
        rkllm_param.temperature = 0.8
        rkllm_param.repeat_penalty = 1.2
        rkllm_param.frequency_penalty = 0.0
        rkllm_param.presence_penalty = 0.0

        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1


        #额外加的
        rkllm_param.num_npu_core = 3

        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init

        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), RKLLMParam, callback_type]
        self.rkllm_init.restype = ctypes.c_int



        self.rkllm_init(ctypes.byref(self.handle), rkllm_param, callback)


        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_void_p]

        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        self.lora_adapter_path = None
        self.lora_model_name = None



    def run(self, prompt):
        # print(f"RKLLM推理线程ID: {threading.get_ident()}, 名称: {threading.current_thread().name}")

        # 构造 prompt 字符串
        formatted_prompt = PROMPT_TEXT_PREFIX + prompt + PROMPT_TEXT_POSTFIX
        formatted_prompt_c = ctypes.c_char_p(formatted_prompt.encode('utf-8'))

        # 调用 rkllm_run，传入 handle, prompt 和 userdata
        self.rkllm_run(self.handle, formatted_prompt_c, None)

    def release(self):
        self.rkllm_destroy(self.handle)



def merge_wav_files(wav_files):
    """确保所有音频片段的格式一致，并合并"""

    if not wav_files:
        return None

    # 路径，直接放在tmp下，放子目录得提前mkdir
    output_wav = f"/tmp/piper_merged_{uuid.uuid4()}.wav"
    audio_segments = []

    for wav_file in wav_files:
        audio = AudioSegment.from_wav(wav_file)
        audio = audio.set_frame_rate(16000)  # 强制相同采样率
        audio = audio.set_channels(2)        #  确保是双声道
        audio_segments.append(audio)

    merged_audio = sum(audio_segments)  # 拼接音频
    merged_audio.export(output_wav, format="wav")  # 保存最终音频

    # 清理临时文件，防止占用大量磁盘空间
    for wav_file in wav_files:
        os.remove(wav_file)

    return output_wav




def piper_worker(sentence, output_wav):
    """独立的 Piper 进程"""
    model_path = "/home/rock/piper/GB_female_south/en_GB-southern_english_female-low.onnx"
    piper_path = "/home/rock/piper/piper"

    # 指定 CPU 核心运行  taskset -c 0-7 限制 Piper 运行在 0-7 号 CPU 核心，防止单核 CPU 过载
    cmd = f'echo "{sentence}" | taskset -c 0-7 {piper_path} --model {model_path} --output-file {output_wav}'

    print(f"Running Piper command: {cmd}")

    # 子进程运行 Piper  不用subprocess.run:subprocess.run() 是阻塞式的，即使在子线程运行，它仍然会等待命令执行完成，导致 Piper 运行时整个进程被卡住。
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # 读取 Piper 输出，防止死锁
    # process.wait()	只等待进程结束，但 不会 处理 stdout 和 stderr	适用于 不需要捕获输出 的情况 process.communicate()	等待进程结束，同时 读取stdout和stderr，防止阻塞	适用于 需要获取进程输出，或 避免死锁
    stdout, stderr = process.communicate()

    if process.returncode == 0 and os.path.exists(output_wav):
        return output_wav  # **成功生成音频**
    else:
        print(f"Piper failed: {sentence}\nError: {stderr.decode()}")
        return None

def play_text_with_piper(text):


    # 分割文本
    sentences = [s.strip() for s in re.split(r"[,.!?]", text) if s.strip()]
    print(f"Sentences to process: {sentences}")

    # 创建并行池，限制最大 CPU 负载
    max_workers = min(2, multiprocessing.cpu_count())

    #代替ThreadPoolExecutor 启动太多 Piper 任务，可能会导致系统崩溃。
    with multiprocessing.Pool(processes=max_workers) as pool:
        # 并行执行 Piper 任务,apply_async() 让进程池异步运行
        results = [pool.apply_async(piper_worker, (sentence, f"/tmp/piper_output_{uuid.uuid4()}.wav")) for sentence in sentences]

        # 收集所有音频文件
        wav_files = [r.get() for r in results if r.get() is not None]

    # 合并所有 WAV
    merged_wav = merge_wav_files(wav_files) if wav_files else None

    # 推送到 Piper 队列
    piper_queue.put(merged_wav)



def get_user_input(user_message, history):
     # 确保去掉空格后仍为空
    if not user_message.strip():
        print(f'empty enter:{user_message.strip()}')
        global input_empty
        input_empty=True
        return "",history,gr.update(interactive=True),gr.update(interactive=True)
    print("get_user_input",f"history:{history},user_message:{user_message}")
    #history = history + [[user_message, None]]
    history = history + [{"role": "user", "content": user_message}]
    return "", history,gr.update(interactive=False),gr.update(interactive=False)


# 获取模型输出
def get_RKLLM_output(history):
    global input_empty
    # bug!!! 输入为空时即使返回history(有值)也清屏
    if input_empty:
        print(f'empty enter, current history:{history}')
        input_empty=False
        return history,gr.update(),gr.update(interactive=True), gr.update(interactive=True)


    global global_text, global_state
    global_text = []
    global_state = -1

    print("getrkllmoutput",history[-1]["content"],history)



    # print(f"getrkllmoutput ID: {threading.get_ident()}, name: {threading.current_thread().name}")
    start_time = time.time()
    # 启动模型推理线程
    model_thread = threading.Thread(target=rkllm_model.run, args=(history[-1]["content"],))
    model_thread.start()



    # 初始化当前的对话内容
    history.append({"role": "assistant", "content": ""})


    # 实时更新模型输出
    while model_thread.is_alive() or global_text:
        # print(f"while ID1: {threading.get_ident()}, name: {threading.current_thread().name}")
        if stop_event.is_set():
            print("Stop Inference！")
            stop_event.clear()
            yield history,gr.update(), gr.update(interactive=True), gr.update(interactive=True)
            return
        while global_text:
            # print(f"while ID2: {threading.get_ident()}, name: {threading.current_thread().name}")
            history[-1]["content"] += global_text.pop(0)
            yield history,gr.update(),gr.update(interactive=False), gr.update(interactive=False)


        time.sleep(0.005)
    inference_time = time.time() - start_time
    print(f"RKLLM Inference Time: {inference_time:.2f} seconds")


    final_text = history[-1]["content"]
    play_text_with_piper(final_text)

    # **等待 Piper 语音合成**
    while piper_queue.empty():
        time.sleep(0.1)  # **避免 UI 过度刷新**

    piper_audio = piper_queue.get()  # **从队列获取音频**

    yield history,piper_audio,gr.update(interactive=True), gr.update(interactive=True)








# 加载音频文件函数
def load_audio(filepath):
    # 使用 torchaudio 加载音频文件
    waveform, sample_rate = torchaudio.load(filepath)
    # 将音频采样率调整为 16000Hz（Whisper 的要求）
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)
    return waveform.squeeze(), 16000  # 返回单通道音频和采样率



# 让 Whisper 在子线程中运行，并通过 Queue 与 UI 交互
def audio_to_text(filepath):
    print("Processing audio:", filepath)

    if filepath is None:
        return ""

    # 使用 Queue 存储推理结果
    result_queue = queue.Queue()
    # 用于标记任务是否完成
    transcription_done = threading.Event()

    start_time = time.time()

    # 定义子线程执行的函数
    def whisper_thread():
        audio_data, sample_rate = load_audio(filepath)
        inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
        #  target_language: str - 目标语言 ( "english", "french", "chinese" 等)
        #  task: str - 任务类型 ("transcribe" 转录, "translate" 翻译)
        target_language="english"
        task="transcribe"
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=target_language, task=task)


        with torch.no_grad():
            generated_ids = whisper_model.generate(
                inputs["input_features"],
                forced_decoder_ids=forced_decoder_ids,  # 语言 & 任务
                max_length=256,
                temperature=0.7,
                num_beams=5,
            )
            print(f"Generated Token IDs: {generated_ids}")
            print(f"Generated Token Length: {generated_ids.shape[1]}")

            # print(f"Whisper Thread ID: {threading.get_ident()}, Name: {threading.current_thread().name}")
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"Transcription Result: {transcription}")

            result_queue.put(transcription)  # 结果存入队列
            transcription_done.set()  # 标记任务完成



    #  启动子线程
    thread = threading.Thread(target=whisper_thread, daemon=True)
    thread.start()

    yield "transcribing audio,please wait ..."
    # **主线程不会阻塞，而是每 50ms 检查一次是否有新结果**
    while not transcription_done.is_set() or not result_queue.empty():
        # print(f"audio to text ID 1: {threading.get_ident()}, name: {threading.current_thread().name}")
        while not result_queue.empty():
            # print(f"audio to text ID 2: {threading.get_ident()}, name: {threading.current_thread().name}")
            yield result_queue.get()  # **实时更新 UI**

        # sleep时间越长ui更新变慢，但cpu负担降低  0.05适中
        time.sleep(0.1)  # 避免 UI 过度刷新

    inference_time = time.time() - start_time
    print(f"Whisper Inference Time: {inference_time:.2f} seconds")






if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True, help='Absolute path of the converted RKLLM model on the Linux board;')
    parser.add_argument('--target_platform', type=str, required=True, help='Target platform: e.g., rk3588/rk3576;')
    args = parser.parse_args()

    if not os.path.exists(args.rkllm_model_path):
        print("Error: Please provide the correct rkllm model path, and ensure it is the absolute path on the board.")
        sys.stdout.flush()
        exit()

    if not (args.target_platform in ["rk3588", "rk3576"]):
        print("Error: Please specify the correct target platform: rk3588/rk3576.")
        sys.stdout.flush()
        exit()



    # Fix frequency
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    # Set resource limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # 在程序启动时 **一次性初始化** Whisper 模型
    print("Loading Whisper model...")
     #加载whisper
    # 加载处理器和模型 本地加载，网络加载和网络状况有关
    processor = AutoProcessor.from_pretrained("/home/rock/models/whisper-tiny")
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained("/home/rock/models/whisper-tiny")

    # 使用多核 CPU
    torch.set_num_threads(2)  # 设置多核并行线程数
    device = torch.device("cpu")  # 指定设备为 CPU
    whisper_model = whisper_model.to(device)
    print("Whisper model loaded successfully!")

    # Initialize RKLLM model
    print("=========RKLLM init....===========")
    sys.stdout.flush()
    model_path = args.rkllm_model_path
    rkllm_model = RKLLM(model_path)
    print("RKLLM Model has been initialized successfully！")
    print("==============================")
    sys.stdout.flush()




    # 创建 Gradio 界面
    with gr.Blocks(title="Chat Ptech Server", fill_width=True,css=css) as chatRKLLM:
        gr.Markdown("# Welcome to Gradio whisper-rkllm-piper server")
        with gr.Row():
            # 左侧 Chatbot 区域
            with gr.Column(scale=7, variant="compact"):
                chatbot = gr.Chatbot(label="Chat RKLLM on NPU", height="80vh", type='messages')
                with gr.Row():
                    with gr.Column(scale=9):
                        msg = gr.Textbox(
                            placeholder="Enter your message here and hit return when you're ready...",
                            interactive=True,
                            container=False,
                            autoscroll=True,
                        )
                    with gr.Column(scale=1):
                        stop_btn = gr.Button("Stop", elem_id="stop-btn", variant="secondary")




                with gr.Row():
                    clear = gr.Button(value="Clear Chat Window", elem_id="button")


            # 右侧录音和文件上传区域
            with gr.Column(scale=3):
                gr.Markdown("### Record and Transcribe")

                # 第一行：录音按钮
                with gr.Row():
                    record_btn = gr.Audio( type="filepath", label="Record Audio", sources=['microphone', 'upload'])

                # 第二行：转录文本框
                with gr.Row():
                    transcribed_text = gr.Textbox(label="Transcribed Text", interactive=False, lines=5)

                # 第三行：提交按钮
                with gr.Row():
                    submit_btn = gr.Button("Submit Transcription", elem_id="button")

                # gr.Markdown("### File Upload")

                # # 文件上传部分
                # with gr.Row():
                #     files = gr.Files(
                #         interactive=True,
                #         file_count="multiple",
                #         file_types=["text", ".pdf", ".xlsx", ".py", ".txt", ".dart", ".c", ".jsx", ".xml", ".css", ".cpp",
                #                     ".html", ".docx", ".doc", ".js", ".json", ".csv"],
                #         label="Upload Files"
                #     )
                gr.Markdown("### Speech and Play")
                piper_audio = gr.Audio(label="Piper Speech Output", interactive=False)

        record_btn.change(audio_to_text, record_btn, transcribed_text,concurrency_limit=2, concurrency_id="cpu_queue")

        # 参数1为函数,参数2[]为输入多个参数,参数3[]为输出多个参数
        # 增加 concurrency_limit 并行处理多个用户请求，提高吞吐量。最多同时运行 2 个推理任务，每个任务占2核，最多占4核
        # 给 RKLLM 和 Whisper 分别分配不同的任务队列，确保不会一个任务卡住整个服务器。如果 5 个用户都用 Whisper，最多 3 个任务并行，其余的进入 cpu_queue，而 RKLLM 仍然可以运行 2 个任务。
        submit_btn.click(get_user_input, [transcribed_text, chatbot], [transcribed_text, chatbot,transcribed_text, submit_btn], queue=False).then(get_RKLLM_output, chatbot, [chatbot,piper_audio,transcribed_text, submit_btn],concurrency_limit=2, concurrency_id="npu_queue")
        msg.submit(get_user_input, [msg, chatbot], [msg, chatbot,msg, submit_btn], queue=False).then(get_RKLLM_output, chatbot, [chatbot,piper_audio,msg, submit_btn],concurrency_limit=2, concurrency_id="npu_queue")
       # lambda 函数作用是返回两个空列表，输入为None,输出为chatbot,  queue=False不加入 Gradio 的后台事件队列，而是 立即执行该函数。如果 有多个用户，或者 RKLLM 推理很慢，用户输入可能会 排队等待
        clear.click(lambda: ([]), None, chatbot, queue=False)
        stop_btn.click(stop_RKLLM)


    # 启动事件队列系统，适用于多用户交互
    # 流式输出（yield）功能：get_RKLLM_output 使用 yield 逐步返回内容，如果不启用 queue()，流式响应可能无法正常工作
    # 多个 Gradio 组件的事件并发：如果你的应用有多个事件（比如 Whisper 语音转录 & RKLLM 推理），但没有 queue()，它们可能会相互阻塞
    # Gradio 的 WebSocket 处理机制：对于较长时间运行的任务，如 RKLLM，queue() 允许 Gradio 在 WebSocket 连接上进行更好的管理
    chatRKLLM.queue()
    # 允许整个 Gradio 服务器同时处理 4 个任务（2 个 Whisper 任务 + 2 个 RKLLM 任务）
    # chatRKLLM.queue(default_concurrency_limit=4)

    # Start the Gradio application.
    chatRKLLM.launch(share=True,inbrowser=True)

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    print("====================")


