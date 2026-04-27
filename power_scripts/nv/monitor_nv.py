import pynvml
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import signal
import os

# 設定
INTERVAL = 1.0 

data = []
keep_running = True

def handle_stop(signum, frame):
    global keep_running
    keep_running = False

# 註冊停止信號
signal.signal(signal.SIGTERM, handle_stop)
signal.signal(signal.SIGINT, handle_stop)

def collect_data(log_file, plot_file):
    # 初始化 NVIDIA NVML
    pynvml.nvmlInit()
    
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            print("未偵測到 NVIDIA GPU 請確認驅動程式。")
            return

        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
        print(f"監控中 (NVIDIA)... 發現 {device_count} 張顯卡，寫入至 {log_file}")

        while keep_running:
            ts = time.strftime("%H:%M:%S")
            for i, h in enumerate(handles):
                try:
                    # 獲取功耗 (單位從 milliwatts 轉為 Watts)
                    pwr_info = pynvml.nvmlDeviceGetPowerUsage(h)
                    pwr = pwr_info / 1000.0 if pwr_info is not None else None
                    
                    # 獲取頻率 (單位 MHz)
                    # NVML_CLOCK_GRAPHICS = 0
                    clk = pynvml.nvmlDeviceGetClockInfo(h, 0)
                    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(h)
                    gpu_util = util_info.gpu
                    mem_util = util_info.memory
                    
                    data.append({
                        "Time": ts, 
                        "GPU": f"GPU_{i}", 
                        "MHz": clk, 
                        "Temperature": temp, 
                        "GPU_Utilization": gpu_util,
                        "Memory_Utilization": mem_util,
                        "Watt": pwr
                    })
                except Exception as e:
                    # 避免在 loop 中頻繁噴錯
                    continue
            time.sleep(INTERVAL)

        # 停止後處理數據
        if data:
            df = pd.DataFrame(data)
            df.to_csv(log_file, index=False)
            
            # 繪圖邏輯
            plt.figure(figsize=(10, 5))
            df_plot = df.dropna(subset=["Watt"])
            
            for gpu in df['GPU'].unique():
                subset = df_plot[df_plot['GPU'] == gpu]
                if not subset.empty:
                    plt.plot(subset['Time'], subset['Watt'], label=f'{gpu} Power')
            
            plt.title("NVIDIA GPU Power Profile")
            plt.xlabel("Time")
            plt.ylabel("Power (Watt)")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_file)
            print(f"圖表已儲存至: {plot_file}")
        else:
            print("監控結束，但沒有收集到任何數據。")
            
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_nv.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "gpu_stats.csv")
    plot_file = os.path.join(output_dir, "gpu_performance.png")

    collect_data(log_file, plot_file)
