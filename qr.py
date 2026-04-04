import qrcode

html_link = "file:///C:/Users/Gaurav%20Tiwari/OneDrive/Desktop/FINALLL.html"

qr = qrcode.QRCode(
    version=1,
    box_size=10,
    border=4
)
qr.add_data(html_link)
qr.make(fit=True)

img = qr.make_image(fill="black", back_color="white")
img.save("my_html_qr.png")
