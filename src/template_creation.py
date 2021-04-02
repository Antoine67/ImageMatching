# Cuts random templates from images

import random, os , csv
import cv2 as cv

def create_folder(path):
           try:
               os.makedirs(path)
           except OSError:
               print ("Creation of the directory %s failed" % path)
           else:
               print ("Successfully created the directory %s" % path)

template_sizes = [256] 
tilesPerImage = 10



for t_size in template_sizes:
    
    dx = dy = t_size
    
    INPATH = ('../storage/dataset/dataset1/')
       
    OUTPATH = (f"../storage/dataset/dataset1_templates/{dx}/")
    create_folder(OUTPATH)
    data_img_temp = []
    
    files = os.listdir(INPATH) 
    
    
    for file in files: 
        im = cv.imread(INPATH+file)
        for i in range(1, tilesPerImage+1):
            newname = file.replace('.', '_{:03d}.'.format(i))
            h,w,d = im.shape
            #print(w, h, dx, dy)
            x = random.randint(0, w-dx-1)
            y = random.randint(0, h-dy-1)
            #print("Cropping {}: {},{} -> {},{}".format(file, x,y, x+dx, y+dy))
            
            #print('x1:',x,'y1:',y,'x2:',x+dx,'y2:',y+dy)
            crop_img = im[y:y+dy, x:x+dx]
            cv.imwrite(OUTPATH+newname, crop_img)
            data_img_temp.append([file,
                                  newname,
                                  x, y,
                                  x+dx, y+dy ])
     
    
    
    
    with open(f"../results/template_creation_{t_size}.csv", 'w+', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        #headers
        filewriter.writerow([
                    "image",
                    "template",
                    "x1",
                    "y1",
                    "x2",
                    "y2"
                ])
        
        
        for d in data_img_temp: # filenames
             filewriter.writerow([
                    d[0],
                    d[1],
                    d[2],
                    d[3],
                    d[4],
                    d[5],
                ])
