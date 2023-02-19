import os

main_dir = 'C:\\Users\\ricky\\Downloads\\Skins\\Skins\\'
for index,file in enumerate(os.listdir(main_dir)):
    os.rename(main_dir+file,main_dir+str(index)+'.png')