import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter.messagebox import showinfo
from PIL import ImageTk, Image
import matplotlib.pyplot as plt

import operations

win = tk.Tk()
win.title("Task 1")# Label
win.geometry("600x340")
win.resizable(False, False)

img = ImageTk.PhotoImage(Image.open("1.jpg"))
# img = img.resize((600, 500))
l = Label(image=img)
l.pack()
#BackGround
#
# bg = PhotoImage(file="1.jpg")
# canvas1 = Canvas(win, width=600,height=500)
# canvas1.pack(fill="both", expand=True)
# canvas1.create_image(0, 0, image=bg,anchor="nw")

#Labels
Lbl1 = ttk.Label(win, text="Feature #1")
Lbl1.place(x=60, y=20)

Lbl2 = ttk.Label(win, text="Feature #2")
Lbl2.place(x=470, y=20)

Lbl3 = ttk.Label(win, text="Class #1")
Lbl3.place(x=60, y=100)

Lbl4 = ttk.Label(win, text="Class #2")
Lbl4.place(x=470, y=100)

Lbl5 = ttk.Label(win, text="Enter learning rate:")
Lbl5.place(x=20, y=180)

Lbl6 = ttk.Label(win, text="Enter number of epochs:")
Lbl6.place(x=20, y=220)

Lbl7 = ttk.Label(win, text="Enter the threshold:")
Lbl7.place(x=20, y=260)

#TextFields
txt_eta = tk.StringVar()
eta = ttk.Entry(win, width=15, textvariable=txt_eta)
eta.place(x=180, y=180, width=100)

txt_m = tk.StringVar()
m = ttk.Entry(win, width=15, textvariable=txt_m)
m.place(x=180, y=220, width=100)

txt_thresh = tk.StringVar()
thresh = ttk.Entry(win, width=15, textvariable=txt_thresh)
thresh.place(x=180, y=260, width=100)

#CheckBox
agreement = tk.StringVar()
def agreement_changed():
    print(agreement.get())
    #tk.messagebox.showinfo(title='Result',message=agreement.get())


ttk.Checkbutton(win,
                text='Add bias',
                command=agreement_changed,
                variable=agreement,
                onvalue='add',
                offvalue='dont_add').place(x=20, y=300)

#ComboBox
#Features
selected_feature1 = tk.StringVar()
feature1_cb = ttk.Combobox(win, textvariable=selected_feature1)
feature1_cb['values'] = ("bill_length", "bill_depth", "flipper_length","gender ","body_mass")
feature1_cb['state'] = 'readonly'
feature1_cb.place(x=20, y=50)

selected_feature2 = tk.StringVar()
feature2_cb = ttk.Combobox(win, textvariable=selected_feature2)
feature2_cb['values'] = ("bill_length", "bill_depth", "flipper_length","gender ","body_mass")
feature2_cb['state'] = 'readonly'
feature2_cb.place(x=420, y=50)

#Classes
selected_class1 = tk.StringVar()
class1_cb = ttk.Combobox(win, textvariable=selected_class1)
class1_cb['values'] = ("Adelie", "Gentoo", "Chinstrap")
class1_cb['state'] = 'readonly'
class1_cb.place(x=20, y=130)

selected_class2 = tk.StringVar()
class2_cb = ttk.Combobox(win, textvariable=selected_class2)
class2_cb['values'] = ("Adelie", "Gentoo", "Chinstrap")
class2_cb['state'] = 'readonly'
class2_cb.place(x=420, y=130)

# def class_changed(event):
#     """ handle the changing event """
#     showinfo(
#         title='Result',
#         message=f'You selected {selected_feature1.get()}!'
#     )

feature1_cb.bind('<<ComboboxSelected>>')
feature2_cb.bind('<<ComboboxSelected>>')
class1_cb.bind('<<ComboboxSelected>>')
class2_cb.bind('<<ComboboxSelected>>')

#Button
def submit(event, score):
    showinfo(
        title='Result',
        message=f'The Score =  {score}!'
    )

def click():
    if selected_class1.get() == "" and selected_class2.get() == "":
        print("EMPTY !!")
    else:
        df = operations.preprocessing()
        df_Adelie = df[df['species'] == "Adelie"]
        df_Gentoo = df[df['species'] == "Gentoo"]
        df_Chinstrap = df[df['species'] == "Chinstrap"]
        if str(selected_class1.get()) == "Adelie":
            # print("IIINNNNNNN")
            c1 = df_Adelie
            # print(df[df['species'] == "Adelie"])
        elif str(selected_class1.get()) == "Gentoo":
            c1 = df_Gentoo
        elif str(selected_class1.get()) == "Chinstrap":
            c1 = df_Chinstrap

        if str(selected_class2.get()) == "Adeli1e":
            c2 = df_Adelie
        elif str(selected_class2.get()) == "Gentoo":
            c2 = df_Gentoo
        elif str(selected_class2.get()) == "Chinstrap":
            c2 = df_Chinstrap

        c1_c2 = operations.append_DF(c1, c2)
        if agreement.get() == 'add':
            check = True
        else:
            check = False

        indx1 = feature1_cb.current()
        indx2 = feature2_cb.current()
        ConfMat, score = operations.split_features(c1_c2, indx1, indx2, float(txt_eta.get()), int(txt_m.get()), float(txt_thresh.get()), check)
        showinfo(
            title='Result',
            message=f'The true positive =  {ConfMat[0, 0]}\n'
                    f'the false positive = {ConfMat[0, 1]}\n'
                    f'the false negative = {ConfMat[1, 0]}\n'
                    f'the true negative = {ConfMat[1, 1]}\n'
                    f'the score = {score}'
        )

        #submit(score)

action = ttk.Button(win, text="SUBMIT",width=40,command=click)
action.place(x=200, y=300)
win.mainloop()