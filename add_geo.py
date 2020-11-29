import os, glob
from GPSPhoto import gpsphoto

geodict = {}
geodict["95138"] = (37.256, -121.775)
geodict["75093"] = (33.035, -96.805)
geodict["47907"] = (40.425, -86.916)
geodict["27712"] = (36.088, -78.923)

GPSLatitude = 0
GPSLongitude = 0
GPSLatitudeRef = 'N'
GPSLongitudeRef = 'S'
GPSAltitudeRef = 0
GPSAultitude = 0

list95138 = ['Alice', 'Anubis', 'Belthoff', 'Berkay', 'Blueeye', 'Boo', 'Brownie', 'Bw', 'Caramel', 'burrito', 'camara']
list75093 = ['Cavalier', 'Celine', 'Chester', 'Coco', 'Doug', 'Francis', 'Fritz', 'Gatsby', 'Gummy']
list47907 = ['Gw', 'Haru', 'Henry', 'John', 'Louie', 'Major', 'Marshie', 'Max', 'Maymo', 'Mishka', 'Natia', 'Neo', 'Noodle', 'Oliver', 'Perry']
list27712 = ['Rb', 'Sammie', 'Shepherd', 'Snowy', 'Spitz', 'Summer', 'Teton', 'Tret', 'Utah', 'Watson', 'Weasel', 'Zeus']


def add_all(mybasedir):
    os.chdir(mybasedir)
    for petid in os.listdir("./"):
        if os.path.isdir(mybasedir+petid):
            os.chdir(mybasedir+petid)
            if petid in list95138:
                gps = geodict["95138"]
            elif petid in list75093:
                gps = geodict["75093"]
            elif petid in list47907:
                gps = geodict["47907"]
            elif petid in list27712:
                gps = geodict["27712"]
            for photofile in glob.glob("*.jpg"):
                photo = gpsphoto.GPSPhoto(photofile)
                info = gpsphoto.GPSInfo(gps)
                photo.modGPSData(info, photofile)
            os.chdir(mybasedir)

def add_lost_and_found(mybasedir):
    os.chdir(mybasedir)
    for photofile in glob.glob("*.jpg"):
        petid = photofile.split('_')[0]
        if petid in list95138:
            gps = geodict["95138"]
        elif petid in list75093:
            gps = geodict["75093"]
        elif petid in list47907:
            gps = geodict["47907"]
        elif petid in list27712:
            gps = geodict["27712"]
        photo = gpsphoto.GPSPhoto(photofile)
        info = gpsphoto.GPSInfo(gps)
        photo.modGPSData(info, photofile)
        #photo.stripData(photofile) would strip the GPS
        #data =gpsphoto.getGPSData(photofile) would get the data. 
mybasedir = "/Volumes/Seagate Expansion Drive/Dog_Dataset/Outdoor/NoAug/lost_and_found"
add_lost_and_found(mybasedir)
