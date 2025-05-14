import tkinter as tk
import subprocess
import threading
from tkinter import filedialog, ttk
from tkinter import messagebox, scrolledtext


def open_test_window():
    test_window = tk.Toplevel()
    test_window.title("模型测试")
    test_window.geometry("960x640")
    test_window.resizable(True, True)

    # 设置字体和颜色主题
    font_family = "SimHei"
    bg_color = "#f0f0f0"
    fg_color = "#333333"
    button_bg = "#4CAF50"
    button_fg = "white"

    test_window.configure(bg=bg_color)

    # 创建顶部标题
    title_frame = tk.Frame(test_window, bg=bg_color)
    title_frame.pack(fill=tk.X, padx=20, pady=10)

    title_label = tk.Label(title_frame, text="模型测试配置", font=(font_family, 16, "bold"), bg=bg_color, fg=fg_color)
    title_label.pack(anchor=tk.W)

    # 创建配置区域
    config_frame = tk.LabelFrame(test_window, text="测试参数", font=(font_family, 12), bg=bg_color, fg=fg_color)
    config_frame.pack(fill=tk.X, padx=20, pady=10)

    # 文件选择区域
    file_frame = tk.Frame(config_frame, bg=bg_color)
    file_frame.pack(fill=tk.X, padx=10, pady=10)

    # file_path
    tk.Label(file_frame, text="测试数据路径:", font=(font_family, 10), bg=bg_color, fg=fg_color).grid(row=0, column=0,
                                                                                                      sticky=tk.W,
                                                                                                      pady=5)
    file_path_combobox = ttk.Combobox(file_frame, width=60, font=(font_family, 10))
    file_path_combobox.grid(row=0, column=1, sticky=tk.W, pady=5, padx=10)

    select_file_button = tk.Button(file_frame, text="浏览...",
                                   command=lambda: select_file(file_path_combobox, "NPY files (*.npy)"),
                                   bg=button_bg, fg=button_fg, font=(font_family, 10))
    select_file_button.grid(row=0, column=2, sticky=tk.W, pady=5)

    # 创建输出区域
    output_frame = tk.LabelFrame(test_window, text="测试输出", font=(font_family, 12), bg=bg_color, fg=fg_color)
    output_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    output_text = scrolledtext.ScrolledText(output_frame, font=(font_family, 10), wrap=tk.WORD)
    output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # 创建状态栏
    status_frame = tk.Frame(test_window, bg="#e0e0e0", height=25)
    status_frame.pack(fill=tk.X, side=tk.BOTTOM)

    status_label = tk.Label(status_frame, text="就绪", font=(font_family, 9), bg="#e0e0e0", fg="#666666", anchor=tk.W)
    status_label.pack(fill=tk.X, padx=10, pady=5)

    # 创建按钮区域
    button_frame = tk.Frame(test_window, bg=bg_color)
    button_frame.pack(fill=tk.X, padx=20, pady=10)

    run_button = tk.Button(button_frame, text="开始测试",
                           command=lambda: run_script(file_path_combobox.get(), output_text, status_label),
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
    def run_script(file_path, output_widget, status_widget):
        # 清空输出
        output_widget.delete(1.0, tk.END)

        try:
            # 构建命令
            command = ['python', './npy.py']
            if file_path:
                command.extend(['--file_path', file_path])

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