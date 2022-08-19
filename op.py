from bs4 import BeautifulSoup
import requests
from tkinter import *
from tkinter import ttk
import webbrowser
from time import sleep
from tkinter import Tk

def rankbot_activation():
    u = Username.get()
    name = 'https://na.op.gg/summoners/na/' + u
    headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Mobile Safari/537.36'}
    source = requests.get(name, headers=headers).text
    soup = BeautifulSoup(source, "html.parser")
    wr = soup.find('body').find("div", {"id": "__next"}).find("div", {"class": "css-1kj5las eioz3421"}).find("div", {"class": "league-stats"}).find("div", {"class": "css-chm1ih e1x14w4w0"}).find("div", {"class": "win-lose"}).getText()
    print(str(wr).replace("\t", ""))

master = Tk()
frm = ttk.Frame(master, padding=10)
frm.grid()
c1 = Entry(frm)
c2 = Entry(frm)
c3 = Entry(frm)
c4 = Entry(frm)
c5 = Entry(frm)
c6 = Entry(frm)
c7 = Entry(frm)
c8 = Entry(frm)
c9 = Entry(frm)
c10 = Entry(frm)
champs = [c1, c2, c3, c4, c5,c6, c7, c8, c9, c10]
Label(master, text="C1").grid(row=0, column=0)
Label(master, text="C2").grid(row=1, column=0)
Label(master, text="C3").grid(row=2, column=0)
Label(master, text="C4").grid(row=3, column=0)
Label(master, text="C5").grid(row=4, column=0)
Label(master, text="C6").grid(row=0, column=2)
Label(master, text="C7").grid(row=1, column=2)
Label(master, text="C8").grid(row=2, column=2)
Label(master, text="C9").grid(row=3, column=2)
Label(master, text="C10").grid(row=4, column=2)
for i in range(5):
    champs[i].grid(row=i, column=1)
for i in range(5):
    champs[i+5].grid(row=i, column=3)
summoners = Entry(frm)
summoners.grid(row=5)
Button(frm, text='Quit', command=master.quit).grid(row=6, column=0, sticky=W, pady=4)
Button(frm, text='Activate webbot', command=rankbot_activation).grid(row=6, column=1, sticky=W, pady=4)
mainloop()