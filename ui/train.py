import tkinter as tk
import subprocess
import threading
from tkinter import filedialog, ttk
from tkinter import messagebox, scrolledtext


def open_train_window():
    train_window = tk.Toplevel()
    train_window.title("模型训练")
    train_window.geometry("960x640")
    train_window.resizable(True, True)

    # 设置字体和颜色主题
    font_family = "SimHei"  # 使用支持中文的字体
    bg_color = "#f0f0f0"
    fg_color = "#333333"
    button_bg = "#4CAF50"
    button_fg = "white"

    train_window.configure(bg=bg_color)

    # 创建顶部标题
    title_frame = tk.Frame(train_window, bg=bg_color)
    title_frame.pack(fill=tk.X, padx=20, pady=10)

    title_label = tk.Label(title_frame, text="模型训练配置", font=(font_family, 16, "bold"), bg=bg_color, fg=fg_color)
    title_label.pack(anchor=tk.W)

    # 创建配置区域
    config_frame = tk.LabelFrame(train_window, text="训练参数", font=(font_family, 12), bg=bg_color, fg=fg_color)
    config_frame.pack(fill=tk.X, padx=20, pady=10)

    # 文件选择区域
    file_frame = tk.Frame(config_frame, bg=bg_color)
    file_frame.pack(fill=tk.X, padx=10, pady=10)

    # root_path
    tk.Label(file_frame, text="训练数据路径:", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=0, column=0,
                                                                                                  sticky=tk.W,
                                                                                                  pady=5)
    root_path_combobox = ttk.Combobox(file_frame, width=60, font=(font_family, 10))
    root_path_combobox.grid(row=0, column=1, sticky=tk.W, pady=5, padx=10)
    root_path_combobox.set('./dateset/')  # 设置默认值

    select_root_button = tk.Button(file_frame, text="浏览...",
                                   command=lambda: select_file(root_path_combobox, "CSV files (*.csv)"),
                                   bg=button_bg, fg=button_fg, font=(font_family, 10))
    select_root_button.grid(row=0, column=2, sticky=tk.W, pady=5)

    # data_path
    tk.Label(file_frame, text="数据文件路径:", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=1, column=0,
                                                                                                  sticky=tk.W,
                                                                                                  pady=5)
    data_path_combobox = ttk.Combobox(file_frame, width=60, font=(font_family, 10))
    data_path_combobox.grid(row=1, column=1, sticky=tk.W, pady=5, padx=10)
    data_path_combobox.set('KNN_3_month.csv')  # 设置默认值

    select_data_button = tk.Button(file_frame, text="浏览...",
                                   command=lambda: select_file(data_path_combobox, "CSV files (*.csv)"),
                                   bg=button_bg, fg=button_fg, font=(font_family, 10))
    select_data_button.grid(row=1, column=2, sticky=tk.W, pady=5)

    # 新增：weather_root_path
    tk.Label(file_frame, text="气象数据根路径:", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=2, column=0,
                                                                                                  sticky=tk.W,
                                                                                                  pady=5)
    weather_root_path_combobox = ttk.Combobox(file_frame, width=60, font=(font_family, 10))
    weather_root_path_combobox.grid(row=2, column=1, sticky=tk.W, pady=5, padx=10)
    weather_root_path_combobox.set('./weather/')  # 设置默认值

    select_weather_root_button = tk.Button(file_frame, text="浏览...",
                                          command=lambda: select_file(weather_root_path_combobox, "Directory"),
                                          bg=button_bg, fg=button_fg, font=(font_family, 10))
    select_weather_root_button.grid(row=2, column=2, sticky=tk.W, pady=5)

    # 新增：weather_data_path
    tk.Label(file_frame, text="气象数据文件:", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=3, column=0,
                                                                                                  sticky=tk.W,
                                                                                                  pady=5)
    weather_data_path_combobox = ttk.Combobox(file_frame, width=60, font=(font_family, 10))
    weather_data_path_combobox.grid(row=3, column=1, sticky=tk.W, pady=5, padx=10)
    weather_data_path_combobox.set('weather.csv')  # 设置默认值

    select_weather_data_button = tk.Button(file_frame, text="浏览...",
                                          command=lambda: select_file(weather_data_path_combobox, "CSV files (*.csv)"),
                                          bg=button_bg, fg=button_fg, font=(font_family, 10))
    select_weather_data_button.grid(row=3, column=2, sticky=tk.W, pady=5)

    # 模型参数区域
    param_frame = tk.Frame(config_frame, bg=bg_color)
    param_frame.pack(fill=tk.X, padx=10, pady=10)

    # 左侧参数
    left_param_frame = tk.Frame(param_frame, bg=bg_color)
    left_param_frame.pack(side=tk.LEFT, padx=5)

    tk.Label(left_param_frame, text="模型ID:", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=0, column=0,
                                                                                                  sticky=tk.W,
                                                                                                  pady=5)
    id_len_entry = tk.Entry(left_param_frame, width=20, font=(font_family, 10))
    id_len_entry.grid(row=0, column=1, sticky=tk.W, pady=5)

    tk.Label(left_param_frame, text="输入维度(enc_in):", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=1,
                                                                                                              column=0,
                                                                                                              sticky=tk.W,
                                                                                                              pady=5)
    enc_in_entry = tk.Entry(left_param_frame, width=20, font=(font_family, 10))
    enc_in_entry.grid(row=1, column=1, sticky=tk.W, pady=5)

    tk.Label(left_param_frame, text="输出维度(dec_in):", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=2,
                                                                                                              column=0,
                                                                                                              sticky=tk.W,
                                                                                                              pady=5)
    dec_in_entry = tk.Entry(left_param_frame, width=20, font=(font_family, 10))
    dec_in_entry.grid(row=2, column=1, sticky=tk.W, pady=5)

    # 右侧参数
    right_param_frame = tk.Frame(param_frame, bg=bg_color)
    right_param_frame.pack(side=tk.RIGHT, padx=5)

    tk.Label(right_param_frame, text="预测维度(c_out):", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=0,
                                                                                                              column=0,
                                                                                                              sticky=tk.W,
                                                                                                              pady=5)
    c_out_entry = tk.Entry(right_param_frame, width=20, font=(font_family, 10))
    c_out_entry.grid(row=0, column=1, sticky=tk.W, pady=5)

    tk.Label(right_param_frame, text="序列长度(seq_len):", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=1,
                                                                                                                column=0,
                                                                                                                sticky=tk.W,
                                                                                                                pady=5)
    seq_len_entry = tk.Entry(right_param_frame, width=20, font=(font_family, 10))
    seq_len_entry.grid(row=1, column=1, sticky=tk.W, pady=5)

    tk.Label(right_param_frame, text="预测长度(pred_len):", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(
        row=2, column=0, sticky=tk.W, pady=5)
    pred_len_entry = tk.Entry(right_param_frame, width=20, font=(font_family, 10))
    pred_len_entry.grid(row=2, column=1, sticky=tk.W, pady=5)

    # 新增：itr
    tk.Label(right_param_frame, text="实验次数(itr):", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(
        row=3, column=0, sticky=tk.W, pady=5)
    itr_entry = tk.Entry(right_param_frame, width=20, font=(font_family, 10))
    itr_entry.grid(row=3, column=1, sticky=tk.W, pady=5)
    itr_entry.insert(0, "3")  # 设置默认值

    # 新增：train_epochs
    tk.Label(left_param_frame, text="训练轮数(train_epochs):", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(
        row=3, column=0, sticky=tk.W, pady=5)
    train_epochs_entry = tk.Entry(left_param_frame, width=20, font=(font_family, 10))
    train_epochs_entry.grid(row=3, column=1, sticky=tk.W, pady=5)
    train_epochs_entry.insert(0, "10")  # 设置默认值

    # 新增：batch_size
    tk.Label(right_param_frame, text="批量大小(batch_size):", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(
        row=4, column=0, sticky=tk.W, pady=5)
    batch_size_entry = tk.Entry(right_param_frame, width=20, font=(font_family, 10))
    batch_size_entry.grid(row=4, column=1, sticky=tk.W, pady=5)
    batch_size_entry.insert(0, "32")  # 设置默认值

    # 新增：patience
    tk.Label(left_param_frame, text="提前停止耐心值(patience):", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(
        row=4, column=0, sticky=tk.W, pady=5)
    patience_entry = tk.Entry(left_param_frame, width=20, font=(font_family, 10))
    patience_entry.grid(row=4, column=1, sticky=tk.W, pady=5)
    patience_entry.insert(0, "2")  # 设置默认值

    # 创建输出区域
    output_frame = tk.LabelFrame(train_window, text="训练输出", font=(font_family, 12), bg=bg_color, fg=fg_color)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    output_text = scrolledtext.ScrolledText(output_frame, font=(font_family, 10), wrap=tk.WORD)
    output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 创建状态栏
    status_frame = tk.Frame(train_window, bg="#e0e0e0", height=25)
    status_frame.pack(fill=tk.X, side=tk.BOTTOM)

    status_label = tk.Label(status_frame, text="就绪", font=(font_family, 9), bg="#e0e0e0", fg="#666666", anchor=tk.W)
    status_label.pack(fill=tk.X, padx=10, pady=5)

    # 创建按钮区域
    button_frame = tk.Frame(train_window, bg=bg_color)
    button_frame.pack(fill=tk.X, padx=20, pady=10)

    run_button = tk.Button(button_frame, text="开始训练", command=lambda: run_script(
        root_path_combobox.get(), data_path_combobox.get(), weather_root_path_combobox.get(),
        weather_data_path_combobox.get(), id_len_entry.get(),
        enc_in_entry.get(), dec_in_entry.get(), c_out_entry.get(),
        seq_len_entry.get(), pred_len_entry.get(), itr_entry.get(),
        train_epochs_entry.get(), batch_size_entry.get(), patience_entry.get(),
        output_text, status_label),
                           bg="#2196F3", fg="white", font=(font_family, 12), height=1, width=15)
    run_button.pack(side=tk.RIGHT, padx=5)

    clear_button = tk.Button(button_frame, text="清空输出", command=lambda: output_text.delete(1.0, tk.END),
                             bg="#f44336", fg="white", font=(font_family, 10), height=1)
    clear_button.pack(side=tk.RIGHT, padx=5)

    # 文件选择函数
    def select_file(combobox, file_types):
        if file_types == "Directory":
            file_path = filedialog.askdirectory()
        else:
            file_path = filedialog.askopenfilename(filetypes=[(file_types, file_types.split("(")[1].replace(")", ""))])
        if file_path:
            combobox.set(file_path)

    # 运行脚本函数
    def run_script(root_path, data_path, weather_root_path, weather_data_path, model_id,
                   enc_in, dec_in, c_out, seq_len, pred_len, itr,
                   train_epochs, batch_size, patience, output_widget,
                   status_widget):
        # 清空输出
        output_widget.delete(1.0, tk.END)

        try:
            # 构建命令
            command = ['python', './run.py', '--is_training', '1']
            if root_path:
                command.extend(['--root_path', root_path])
            if data_path:
                command.extend(['--data_path', data_path])
            if weather_root_path:
                command.extend(['--weather_root_path', weather_root_path])
            if weather_data_path:
                command.extend(['--weather_data_path', weather_data_path])
            if model_id:
                command.extend(['--model_id', model_id])
            if enc_in:
                command.extend(['--enc_in', enc_in])
            if dec_in:
                command.extend(['--dec_in', dec_in])
            if c_out:
                command.extend(['--c_out', c_out])
            if seq_len:
                command.extend(['--seq_len', seq_len])
            if pred_len:
                command.extend(['--pred_len', pred_len])
            if itr:
                command.extend(['--itr', itr])
            if train_epochs:
                command.extend(['--train_epochs', train_epochs])
            if batch_size:
                command.extend(['--batch_size', batch_size])
            if patience:
                command.extend(['--patience', patience])

            # 更新状态
            status_widget.config(text="正在运行...")
            output_widget.insert(tk.END, f"执行命令: {' '.join(command)}\n\n")

            # 启动子进程
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')

            # 启动监控线程
            def monitor_output():
                def insert_line(line):
                    output_widget.insert(tk.END, line)
                    output_widget.see(tk.END)

                for line in process.stdout:
                    output_widget.after(0, insert_line, line)

                process.wait()

                # 更新状态
                if process.returncode == 0:
                    status_text = "执行完成"
                    status_color = "#4CAF50"
                else:
                    status_text = f"执行失败，返回码: {process.returncode}"
                    status_color = "#f44336"

                output_widget.after(0, insert_line, f"\n{status_text}\n")
                status_widget.after(0, lambda: status_widget.config(text=status_text, fg=status_color))

            output_monitor = threading.Thread(target=monitor_output)
            output_monitor.daemon = True
            output_monitor.start()

        except Exception as e:
            error_msg = f"发生错误: {str(e)}"
            output_widget.insert(tk.END, error_msg)
            status_widget.config(text=error_msg, fg="#f44336")
            messagebox.showerror("错误", error_msg)
