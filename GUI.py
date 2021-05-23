import tkinter as tk

top =  tk.Tk()
B =  tk.Button(top, text ="Hello")
canvas = tk.Canvas(width=1000, height=1000,bg='black')
canvas.pack()
B.pack()
top.mainloop()