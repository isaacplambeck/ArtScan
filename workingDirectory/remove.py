import tkinter as tk
import sys

def testprint():
    print("other function")

def redirect_output():
    class StdoutRedirector:
        def __init__(self, text_widget):
            self.text_space = text_widget

        def write(self, message):
            self.text_space.insert(tk.END, message)

    root = tk.Tk()
    root.title("Console Output")
    
    text_area = tk.Text(root, wrap=tk.WORD)
    text_area.pack(expand=True, fill=tk.BOTH)

    # Redirect stdout to the text widget
    sys.stdout = StdoutRedirector(text_area)

    def print_message():
        print("This is a printed message.")

    button = tk.Button(root, text="Print Message", command=testprint)
    button.pack()

    root.mainloop()

# Run the GUI application
redirect_output()
