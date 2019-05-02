from urllib.request import urlopen
import argparse
import requests as req
from bs4 import BeautifulSoup


# Target URL https://www.google.com/search?biw=952&bih=995&tbm=isch&sa=1&ei=wU_KXOy3KJK0mAXN17e4Cw&q=dog&oq=dog&gs_l=
# img.3..0l10.21772.24613..24753...0.0..0.98.846.9......1....1..gws-wiz-img.....0..0i24j35i39.-wnrvv8VOo8
dog = "dog"
url_info = "https://www.google.com/search?biw=952&bih=995&tbm=isch&sa=1&ei=JoPKXOaTIciRr7wP4_udgAQ&q=dog&oq=dog&gs_l=img.3..35i39j0l9.3678812.3679457..3679591...0.0..0.100.285.2j1......1....1..gws-wiz-img.RPDez6zEFSo"

params = {
    "q" : dog,
    "tbm" : "isch",
    "gs_l" : "img.3..35i39j0l9.3678812.3679457..3679591...0.0..0.100.285.2j1......1....1..gws-wiz-img.RPDez6zEFSo",
}

html_object = req.get(url_info, params)
if html_object.status_code == 200:
    bs_object = BeautifulSoup(html_object.text, "html.parser")
    img_data = bs_object.find_all("img")

    for i in enumerate(img_data[1:]):
        t = urlopen(i[1].attrs['src']).read()
        filename = "Dog_" + str(i[0]+1)+'.jpg'

        with open(filename, "wb") as f:
            f.write(t)

        print("Img Save Complete")


