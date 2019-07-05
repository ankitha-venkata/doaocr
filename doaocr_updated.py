import pprint
import re
import PIL
from PIL import Image #https://pillow.readthedocs.io/en/3.1.x/reference/Image.html
import pytesseract
import numpy as np
from scipy import misc
import pandas as pd


class GeneralForm():
    def __init__(self, image_path):
        self.image_path         = image_path
        self.bounding_box_dict  =   {
            #format of pixel dimensions to specify the field box boundaries: ( x_min, y_min, x_max, y_max ) 
            #http://nicodjimenez.github.io/boxLabel/annotate.html
            "Date"   :                                  (708,131,1084,200),
            "Purpose"   :                               (939,534,1091,632),
            "Name"     :                                (170,703,1106,784),
            "Krishicard":                               (562,870,1100,942),
            "Survey"    :                               (708,1017,1099,1085),
            "Area"      :                               (707,1165,1083,1232),
            "Village"   :                               (707,1335,1096,1403)
            
        }

        self.component_contents_dict = dict(zip(self.bounding_box_dict.keys(), len(self.bounding_box_dict) * [""])) #someone add comment

    def reading_boundaries(self):
        image = Image.open(self.image_path)
        for form_field, bounding_box in self.bounding_box_dict.items():
            # Contrast of the image is maxed out
            x1, y1, x2, y2  = bounding_box
            xx              = (x2-x1) << 2 #Bitwise left shift
            yy              = (y2-y1) << 2 
            the_crop        = image.crop(bounding_box) #crops according to pixel size of each item
            the_crop        = the_crop.resize((xx,yy),PIL.Image.LANCZOS) #scales cropped part up and it's resampled with a high quality downsampling filter
            area            = (xx * yy)
            gray            = the_crop.convert('L') #converts to black and white
            bw              = np.asarray(gray).copy() #converts the gray list into an array
            bw[bw  < 200]   = 0
            bw[bw >= 200]   = 255
            the_crop        = misc.toimage(bw) #takes array and returns a PIL image

            if "Checkbox" in form_field:
                #checks if the box has been filled by checking if more than 50% of its area has been filled
                filled_area=np.sum(bw)/256
                check=filled_area<=(0.1*area)
                self.component_contents_dict[form_field]=check
            else:
                self.component_contents_dict[form_field] = self.cleanup(pytesseract.image_to_string(the_crop))
        output=self.component_contents_dict[form_field]

        
    def cleanup(self, st):
        """character cleanup for common/repeatable OCR problems"""
        st = re.sub('!', '1', st) #https://docs.python.org/2/library/re.html
        st = re.sub(r'(\d) (\d)', r'\1\2', st)
        st = re.sub(r'\n|\r',' ', st)
        return st


    def __repr__(self):
        #returns the well formatted version of the data
        output=self.component_contents_dict
        df=pd.DataFrame(output, index=['a']) 
        #adds header only if a new file is being created
        try:
            file_test=open('newer_csv_file.csv')
            file_test.close()
            df.to_csv('newer_csv_file.csv',mode='a',index=False, header=False)
        except IOError:
            df.to_csv('newer_csv_file.csv',mode='a',index=False, header=True)
        return pprint.pformat(self.component_contents_dict)


    @classmethod
    def edges(cls):
        from scipy import ndimage, misc
        import numpy as np
        from skimage import feature
        original_img = Image.open("newtrial_wobox_pg1.jpg")
        gray = original_img.convert('L')

        # Converts pixels to pure black or white
        bw = np.asarray(gray).copy()

        # Range of pixels is 0-255
        bw[bw < 245]  = 0 # Black
        bw[bw >= 245] = 255 # White
        bw[bw == 0] = 254
        bw[bw == 255] = 0
        im = bw
        im = ndimage.gaussian_filter(im, 1) #Multidimensional gaussian filter https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter
        edges_canny = feature.canny(im, sigma=2) #detects edges using canny method https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
        labels, numobjects =ndimage.label(im)
        slices = ndimage.find_objects(labels)
        print('\n'.join(map(str, slices)))
        misc.imsave('newtrial_wobox_neg.jpg', im)
        return

        sobel_x = ndimage.sobel(im, axis=0, mode='constant') #detected edges using sobel method
        sobel_y = ndimage.sobel(im, axis=1, mode='constant')
        sobel= np.hypot(sobel_x, sobel_y)
        misc.imsave('newtrial_wobox_neg.jpg', edges_canny)

if __name__ == "__main__":
    read = GeneralForm('scan0012.jpg')
    read.reading_boundaries()
    print(read)
    print(type(read))
