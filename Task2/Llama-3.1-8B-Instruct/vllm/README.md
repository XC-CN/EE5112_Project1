# Llama-3.1-8B-Instruct 鈥?Transformers + vLLM 閮ㄧ讲鎸囧崡

鏈洰褰曟彁渚涘湪鏀寔 CUDA 鐨?Linux/WSL 鐜涓嬶紝浣跨敤 **Hugging Face Transformers** 涓?**vLLM** 閮ㄧ讲 `meta-llama/Llama-3.1-8B-Instruct` 鐨勫畬鏁磋剼鏈拰閰嶇疆銆傜浉姣?`llama.cpp` 鏂规锛岃娴佺▼鐩存帴鍔犺浇瀹樻柟 `.safetensors` 鏉冮噸锛岃幏寰楁洿楂樼簿搴︿笌鍚炲悙銆?

> 鈿狅笍 **鎿嶄綔绯荤粺鎻愰啋**锛歷LLM 瀹樻柟鐩墠浠呮敮鎸?Linux銆傝嫢鍦?Windows 涓绘満锛岃閫氳繃 [WSL2](https://learn.microsoft.com/windows/wsl/install) 鎴栬繙绋?Linux 鏈嶅姟鍣ㄦ墽琛屼互涓嬫楠ゃ€?

## 鐜鍑嗗

1. **鍒涘缓 Conda / Python 铏氭嫙鐜**锛堝缓璁?Python 3.10+锛夛細

   ```bash
   conda create -n llama31-vllm -y
   conda activate llama31-vllm
   ```
2. **瀹夎 CUDA 鍙婇┍鍔?*锛氱‘淇濇樉鍗￠┍鍔ㄤ笌 CUDA 鐗堟湰婊¤冻 vLLM 瑕佹眰锛?= CUDA 12.1锛夈€?
3. **瀹夎渚濊禆**锛?

   ```bash
   pip install -r requirements.txt
   ```

   鍏朵腑鍖呭惈 `transformers`銆乣vllm`銆乣torch`銆乣accelerate` 绛夊簱銆傞娆″畨瑁呰€楁椂杈冮暱锛屽彲寮€鍚浗鍐呴暅鍍忔垨 `pip --extra-index-url` 鍔犻€熴€?

## Hugging Face 璁块棶鍑瘉

1. 鍦ㄦ祻瑙堝櫒鐧诲綍 [Hugging Face](https://huggingface.co/)銆?
2. 璁块棶妯″瀷涓婚〉 [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 鎺ュ彈鍗忚骞剁敵璇疯闂€?
3. 鍒?[Tokens 椤甸潰](https://huggingface.co/settings/tokens) 鐢熸垚 **Read** 鏉冮檺鐨勪釜浜轰护鐗屻€?
4. 鍦?Shell 涓鍑虹幆澧冨彉閲忥細

   ```bash
   export HF_TOKEN="hf_your_read_token"
   ```

   Windows PowerShell (WSL 浠ュ) 鍙娇鐢細`$env:HF_TOKEN="hf_xxx"`銆?

## 杩愯瀵硅瘽绯荤粺

1. 鍒囨崲鍒版湰鐩綍锛?

   ```bash
   cd Task2/Llama-3.1-8B-Instruct/vllm
   ```
2. 鍚姩浜や簰寮忓璇濓細

   ```bash
   python dialogue_system_vllm.py --config config.json
   ```

   - `--cache-dir`锛氳嚜瀹氫箟妯″瀷缂撳瓨鐩綍锛堥粯璁?`./hf_cache`锛夈€?
   - `--no-save`锛氱鐢ㄥ璇濊褰曚繚瀛樸€?
   - `--max-new-tokens`锛氫复鏃惰鐩栫敓鎴愭渶澶ч暱搴︺€?
3. 鍛戒护琛屼氦浜掞細

   - 鏅€氳緭鍏ワ細鐢熸垚鍥炵瓟
   - `stats`锛氭煡鐪嬫ā鍨嬩笌浼氳瘽缁熻
   - `clear`锛氭竻闄や笂涓嬫枃璁板繂
   - `exit`锛氶€€鍑?

瀵硅瘽鍘嗗彶榛樿鍐欏叆 `conversations_vllm/`锛屽彲鍦ㄦ姤鍛婁腑寮曠敤銆?

## 妯″瀷缂撳瓨浣嶇疆

棣栨杩愯浼氳嚜鍔ㄤ粠 Hugging Face 涓嬭浇 `.safetensors` 鏉冮噸涓庡垎璇嶅櫒锛屽苟缂撳瓨鍦?`hf_cache/` 鐩綍锛堝彲閫氳繃 `--cache-dir` 淇敼锛夈€傛缂撳瓨鍙法鑴氭湰澶嶇敤銆?

## 甯歌闂

| 闂                            | 瑙ｅ喅鏂规                                                                                           |
| ------------------------------- | -------------------------------------------------------------------------------------------------- |
| 鎶ラ敊 "vLLM only supports Linux" | 璇峰湪 Linux/WSL2 鐜鎵ц銆?                                                                        |
| CUDA OOM                        | 闄嶄綆 `config.json` 涓殑 `gpu_memory_utilization` 鎴栬缃?`tensor_parallel_size > 1`锛堝鍗★級銆?|
| Token 403 / 404                 | 纭宸插湪妯″瀷椤甸潰鎺ュ彈璁稿彲锛屽苟姝ｇ‘璁剧疆 `HF_TOKEN`銆?                                               |
| 鐢熸垚缂撴參                        | 閫傚綋澧炲ぇ `sampling.top_p/top_k` 鎴栭檷浣?`max_tokens`锛屼繚鎸?GPU 椋庢墖姝ｅ父銆?                      |

## 鐩綍缁撴瀯

```
vllm/
鈹溾攢鈹€ config.json                # 妯″瀷銆侀噰鏍枫€佷細璇濋厤缃?
鈹溾攢鈹€ dialogue_system_vllm.py    # 浜や簰寮忓璇濊剼鏈?
鈹溾攢鈹€ requirements.txt           # Transformers+vLLM 渚濊禆
鈹溾攢鈹€ README.md                  # 鏈鏄?
鈹溾攢鈹€ hf_cache/                  # (杩愯鍚庣敓鎴? 妯″瀷缂撳瓨
鈹斺攢鈹€ conversations_vllm/        # (杩愯鍚庣敓鎴? 瀵硅瘽璁板綍
```

## 鍚庣画鎵╁睍

- 灏?`dialogue_system_vllm.py` 鍖呰涓?RESTful API锛岀剨鎺ュ墠绔垨鏈哄櫒浜烘帶鍒舵帴鍙ｃ€?
- 浣跨敤 [vLLM HTTP Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) 鍦ㄦ湰鍦版彁渚?OpenAI 鍏煎 API銆?
- 瀵规瘮 `llama.cpp` 鏂规涓?vLLM 鐨勫欢杩熴€佸悶鍚愪笌鏄惧瓨浣跨敤锛屼负璇剧▼鎶ュ憡鎻愪緵鏁版嵁鏀拺銆?

> 鑻ラ渶鑷姩鍖栬剼鏈垨 Dockerfile锛屽彲浠ュ湪姝ゅ熀纭€涓婄户缁墿灞曪紝鎴戜篃鍙崗鍔╄ˉ鍏呫€傜閮ㄧ讲椤哄埄锛?
