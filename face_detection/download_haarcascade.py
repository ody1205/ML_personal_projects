import urllib.request as req

website = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalcatface.xml'
savefile = 'haarcascade_frontalface_alt.xml'
req.urlretrieve(website, savefile)