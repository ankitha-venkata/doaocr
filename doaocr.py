import pprint
import re
import PIL
from PIL import Image #https://pillow.readthedocs.io/en/3.1.x/reference/Image.html
import pytesseract
import numpy as np
from scipy import misc
import pandas as pd
import xlwt
from xlwt import Workbook


class GeneralForm():
    """Peforms OCR on the form"""
    def __init__(self, image_path):
        self.image_path         = image_path
        self.bounding_box_dict  =   {
            #                                 key: ( w-start, h-start, w-end, h-end )
            "Date"   :                                  (708,131,1084,200),
            "Checkbox_Purchase_of_traps":               (193,369,203,378),
            "Checkbox_Purchase_of_weedicide":           (192,402,204,413),
            "Checkbox_Construction_of_biogas":          (193,436,204,447),
            "Checkbox_Construction_of_compost":         (192,470,204,480),
            "Checkbox_Purchase_of_micronutrients":      (192,502,204,514),
            "Checkbox_Organic_input_assistance":        (193,537,204,547),
            "Checkbox_Mechanized_paddy_transplanting":  (192,570,205,581),
            "Name"     :                                (155,711,1090,786),
            "Krishicard":                               (562,873,1085,943),
            "Survey"    :                               (708,1020,1084,1086),
            "Area"      :                               (707,1165,1083,1232),
            "Village"   :                               (707,1335,1085,1403),
            
        }

        self.component_contents_dict = dict(zip(self.bounding_box_dict.keys(), len(self.bounding_box_dict) * [""]))

    def parse(self):
        """runs each mask(crop) across the image file to improve OCR functionality"""
        image = Image.open(self.image_path)
        for form_field, bounding_box in self.bounding_box_dict.items():
            # the crops are scaled up and the contrast maxed out in order to enhance character
            # features and increase OCR success
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

            # use this to check out a particular mask
            #if "box_c_address_city_town_zip_postal_code" is form_field:
            #    the_crop.show()

            if "Checkbox" in form_field:
                # a box is considered checked if 10% or more of it's area is black
                temp=np.sum(bw)/256
                checktwo=temp<=(0.5*area)
                self.component_contents_dict[form_field]=checktwo
            else:
                self.component_contents_dict[form_field] = self.clean_text(pytesseract.image_to_string(the_crop))
        optemp=self.component_contents_dict[form_field]
        #print(optemp)
        #pd.DataFrame(optemp)
        #df.to_csv('new_csv_file.csv')

        
    def clean_text(self, st):
        """character cleanup for common/repeatable OCR problems"""
        st = re.sub('!', '1', st) #https://docs.python.org/2/library/re.html
        st = re.sub(r'(\d) (\d)', r'\1\2', st)
        st = re.sub(r'\n|\r',' ', st)
        return st


    def __repr__(self):
        """returns the pretty formatted version of the image data contents""" 
        optemp2=self.component_contents_dict
        df=pd.DataFrame(optemp2, index=['0']) 
        df.to_csv('new_csv_file.csv',mode='a',header=False,newline='')
        return pprint.pformat(self.component_contents_dict)


    @classmethod
    def edges(cls):
        from scipy import ndimage, misc
        import numpy as np
        from skimage import feature
        col = Image.open("newtrial1.jpg")
        gray = col.convert('L')

        # Let numpy do the heavy lifting for converting pixels to pure black or white
        bw = np.asarray(gray).copy()

        # Pixel range is 0...255, 256/2 = 128
        bw[bw < 245]  = 0    # Black
        bw[bw >= 245] = 255 # White
        bw[bw == 0] = 254
        bw[bw == 255] = 0
        im = bw
        im = ndimage.gaussian_filter(im, 1) #Multidimensional gaussian filter https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter
        edges2 = feature.canny(im, sigma=2) #detects edges using canny method https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
        labels, numobjects =ndimage.label(im)
        slices = ndimage.find_objects(labels)
        print('\n'.join(map(str, slices)))
        misc.imsave('newtrialneg.jpg', im) #
        return

        #im = misc.imread('f990.jpg')
        #im = ndimage.gaussian_filter(im, 8) 
        sx = ndimage.sobel(im, axis=0, mode='constant') #detected edges using sobel method
        sy = ndimage.sobel(im, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        misc.imsave('newtrialneg.jpg', edges2)

if __name__ == "__main__":
    f = GeneralForm('newtrialhand4.jpg')
    f.parse()
    #df=pd.DataFrame(Form990('newtrialhand4.jpg'))
    #df.to_csv('new_csv_file.csv')
    print(f)
    print(type(f))
