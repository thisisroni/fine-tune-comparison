import amdsmi
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


def _get_clock_type():
    # AMD SMI Python binding renamed enum between versions.
    if hasattr(amdsmi, "AmdSmiClkType"):
        return amdsmi.AmdSmiClkType.GFX
    if hasattr(amdsmi, "AmdSmiClockType"):
        return amdsmi.AmdSmiClockType.GFX
    raise RuntimeError("Unsupported amdsmi version: missing clock type enum")


def _parse_clock_mhz(clock_info):
    for key in ("clk", "now", "current"):
        if key in clock_info:
            return clock_info[key]
    raise RuntimeError(f"Unknown clock info format: {clock_info}")


def _to_watt(value):
    if value is None:
        return None

    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("n/a", "na", ""):
            return None
        s = s.replace("watts", "").replace("watt", "").replace("w", "").strip()
        try:
            return float(s)
        except ValueError:
            return None

    if isinstance(value, (int, float)):
        # Some interfaces report microwatts.
        if value > 1_000_000:
            return float(value) / 1_000_000.0
        return float(value)

    return None


def _parse_power_watt(power_info):
    for key in ("average_socket_power", "socket_power", "current_socket_power"):
        watt = _to_watt(power_info.get(key))
        if watt is not None:
            return watt
    return None

def handle_stop(signum, frame):
    global keep_running
    keep_running = False

# 註冊停止信號 (當 Bash 殺掉此程式時觸發)
signal.signal(signal.SIGTERM, handle_stop)
signal.signal(signal.SIGINT, handle_stop)

def collect_data(log_file, plot_file):
    amdsmi.amdsmi_init()
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            print("未偵測到 AMD GPU handle，請確認節點與權限。")
            return

        clock_type = _get_clock_type()
        warned = set()

        print(f"監控中 (MI300)... 寫入至 {log_file}")

        while keep_running:
            ts = time.strftime("%H:%M:%S")
            for i, h in enumerate(handles):
                try:
                    # 相容不同 amdsmi 版本的時脈回傳欄位
                    clock_info = amdsmi.amdsmi_get_clock_info(h, clock_type)
                    clk = _parse_clock_mhz(clock_info)
                    pwr = _parse_power_watt(amdsmi.amdsmi_get_power_info(h))
                    data.append({"Time": ts, "GCD": f"GCD_{i}", "MHz": clk, "Watt": pwr})
                except Exception as e:
                    if i not in warned:
                        print(f"GPU {i} 讀取失敗: {e}")
                        warned.add(i)
                    continue
            time.sleep(INTERVAL)

        # 停止後處理數據
        if data:
            df = pd.DataFrame(data)
            df.to_csv(log_file, index=False)
            # 簡易繪圖
            plt.figure(figsize=(10, 5))
            df_plot = df.dropna(subset=["Watt"])
            if df_plot.empty:
                print("警告：有收集到資料，但無有效功耗值可繪圖。")
            for gcd in df['GCD'].unique():
                subset = df_plot[df_plot['GCD'] == gcd]
                if not subset.empty:
                    plt.plot(subset['Time'], subset['Watt'], label=f'{gcd} Power')
            plt.title("MI300 Power Profile")
            if not df_plot.empty:
                plt.legend()
            plt.savefig(plot_file)
        else:
            print("監控結束，但沒有收集到任何 GPU 數據。")
    finally:
        amdsmi.amdsmi_shut_down()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "gpu_stats.csv")
    plot_file = os.path.join(output_dir, "gpu_performance.png")

    collect_data(log_file, plot_file)
