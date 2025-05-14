import tkinter as tk
from ui.train import open_train_window
from ui.pred import open_pred_window
from ui.test import open_test_window

root = tk.Tk()
root.title("主界面")
root.geometry("960x640")  # 设置主窗口大小

train_button = tk.Button(root, text="train", command=open_train_window)
train_button.pack(pady=10)

test_button = tk.Button(root, text="test", command=open_test_window)
test_button.pack(pady=10)

pred_button = tk.Button(root, text="pred", command=open_pred_window)
pred_button.pack(pady=10)

root.mainloop()