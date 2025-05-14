import tkinter as tk
import subprocess
import threading
from tkinter import filedialog, ttk
from tkinter import messagebox, scrolledtext


def open_pred_window():
    pred_window = tk.Toplevel()
    pred_window.title("模型预测")
    pred_window.geometry("960x640")
    pred_window.resizable(True, True)

    # 设置字体和颜色主题
    font_family = "SimHei"
    bg_color = "#f0f0f0"
    fg_color = "#333333"
    button_bg = "#4CAF50"
    button_fg = "white"

    pred_window.configure(bg=bg_color)

    # 创建顶部标题
    title_frame = tk.Frame(pred_window, bg=bg_color)
    title_frame.pack(fill=tk.X, padx=20, pady=10)

    title_label = tk.Label(title_frame, text="模型预测配置", font=(font_family, 16, "bold"), bg=bg_color, fg=fg_color)
    title_label.pack(anchor=tk.W)

    # 创建配置区域
    config_frame = tk.LabelFrame(pred_window, text="预测参数", font=(font_family, 12), bg=bg_color, fg=fg_color)
    config_frame.pack(fill=tk.X, padx=20, pady=10)

    # 文件选择区域
    file_frame = tk.Frame(config_frame, bg=bg_color)
    file_frame.pack(fill=tk.X, padx=10, pady=10)

    # root_path
    tk.Label(file_frame, text="根路径:", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=0, column=0,
                                                                                                sticky=tk.W, pady=5)
    root_path_combobox = ttk.Combobox(file_frame, width=60, font=(font_family, 10))
    root_path_combobox.grid(row=0, column=1, sticky=tk.W, pady=5, padx=10)
    root_path_combobox.set('./dateset_pred/')  # 设置默认值

    select_root_button = tk.Button(file_frame, text="浏览...",
                                   command=lambda: select_file(root_path_combobox, "CSV files (*.csv)"),
                                   bg=button_bg, fg=button_fg, font=(font_family, 10))
    select_root_button.grid(row=0, column=2, sticky=tk.W, pady=5)

    # data_path
    tk.Label(file_frame, text="数据路径:", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=1, column=0,
                                                                                                  sticky=tk.W, pady=5)
    data_path_combobox = ttk.Combobox(file_frame, width=60, font=(font_family, 10))
    data_path_combobox.grid(row=1, column=1, sticky=tk.W, pady=5, padx=10)
    data_path_combobox.set('wu')  # 设置默认值

    select_data_button = tk.Button(file_frame, text="浏览...",
                                   command=lambda: select_file(data_path_combobox, "CSV files (*.csv)"),
                                   bg=button_bg, fg=button_fg, font=(font_family, 10))
    select_data_button.grid(row=1, column=2, sticky=tk.W, pady=5)


    # pth_path
    tk.Label(file_frame, text="模型路径:", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=2, column=0,
                                                                                                  sticky=tk.W, pady=5)
    pth_combobox = ttk.Combobox(file_frame, width=60, font=(font_family, 10))
    pth_combobox.grid(row=2, column=1, sticky=tk.W, pady=5, padx=10)

    select_pth_button = tk.Button(file_frame, text="浏览...",
                                  command=lambda: select_file(pth_combobox, "PTH files (*.pth)"),
                                  bg=button_bg, fg=button_fg, font=(font_family, 10))
    select_pth_button.grid(row=2, column=2, sticky=tk.W, pady=5)

    # 创建输出区域
    output_frame = tk.LabelFrame(pred_window, text="预测输出", font=(font_family, 12), bg=bg_color, fg=fg_color)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    output_text = scrolledtext.ScrolledText(output_frame, font=(font_family, 10), wrap=tk.WORD)
    output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 创建状态栏
    status_frame = tk.Frame(pred_window, bg="#e0e0e0", height=25)
    status_frame.pack(fill=tk.X, side=tk.BOTTOM)

    status_label = tk.Label(status_frame, text="就绪", font=(font_family, 9), bg="#e0e0e0", fg="#666666", anchor=tk.W)
    status_label.pack(fill=tk.X, padx=10, pady=5)

    # 创建按钮区域
    button_frame = tk.Frame(pred_window, bg=bg_color)
    button_frame.pack(fill=tk.X, padx=20, pady=10)

    run_button = tk.Button(button_frame, text="开始预测", command=lambda: run_script(
        root_path_combobox.get(), data_path_combobox.get(), pth_combobox.get(), output_text, status_label),
                           bg="#2196F3", fg="white", font=(font_family, 12), height=1, width=15)
    run_button.pack(side=tk.RIGHT, padx=5)

    clear_button = tk.Button(button_frame, text="清空输出", command=lambda: output_text.delete(1.0, tk.END),
                             bg="#f44336", fg="white", font=(font_family, 10), height=1)
    clear_button.pack(side=tk.RIGHT, padx=5)

    # 文件选择函数
    def select_file(combobox, file_types):
        file_path = filedialog.askopenfilename(filetypes=[(file_types, file_types.split("(")[1].replace(")", ""))])
        if file_path:
            combobox.set(file_path)

    # 运行脚本函数
    def run_script(root_path, data_path, pth, output_widget, status_widget):
        # 清空输出
        output_widget.delete(1.0, tk.END)

        try:
            # 构建命令
            command = ['python', './run.py', '--is_training', '3']
            if root_path:
                command.extend(['--root_path', root_path])
            if data_path:
                command.extend(['--data_path', data_path])
            if pth:
                command.extend(['--pth', pth])

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
