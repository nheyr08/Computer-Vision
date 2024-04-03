import os
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
image_names = ['bunny','star','venus', 'noisy_venus']
import numpy as np
import scipy
import numpy as np
import scipy.sparse as sp
import scipy
import cv2
from PIL import Image
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

class PMSR(object):
    image_col = 0
    image_row = 0
    mask2=[]
    def compute_depth(mask,N):
        """
        compute the depth picture
        """
        im_h, im_w = mask.shape
        N = np.reshape(N, (im_h, im_w, 3))

        # =================get the non-zero index of mask=================
        obj_h, obj_w = np.where(mask != 0)
        no_pix = np.size(obj_h) #37244
        full2obj = np.zeros((im_h, im_w))
        for idx in range(np.size(obj_h)):
            full2obj[obj_h[idx], obj_w[idx]] = idx

        M = scipy.sparse.lil_matrix((2*no_pix, no_pix))
        v = np.zeros((2*no_pix, 1))

        # # ================= fill the M&V =================
        for idx in range(no_pix):
            # obtain the 2D coordinate
            h,p=0,0
            h = obj_h[idx]
            w = obj_w[idx]
            # obtian the surface normal vector
            n_x = N[h, w, 0]
            n_y = N[h, w, 1]
            n_z = N[h, w, 2]
            row_idx = idx * 2
            if mask[h, w+1]:
                idx_horiz = int(full2obj[h, w+1])
                M[row_idx, idx] = -1
                M[row_idx, idx_horiz] = 1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_x / n_z
            elif mask[h, w-1]:
                idx_horiz =int( full2obj[h, w-1])
                M[row_idx, idx_horiz] = -1
                M[row_idx, idx] = 1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_x / n_z

            row_idx = idx * 2 + 1
            if mask[h+1, w]:
                idx_vert = int(full2obj[h+1, w])
                M[row_idx, idx] = 1
                M[row_idx, idx_vert] = -1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_y / n_z
            elif mask[h-1, w]:
                idx_vert = int(full2obj[h-1, w])
                M[row_idx, idx_vert] = 1
                M[row_idx, idx] = -1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_y / n_z

        # =================sloving the linear equations Mz = v=================
        MtM = M.T @ M
        Mtv = M.T @ v
        z = scipy.sparse.linalg.spsolve(MtM, Mtv)

        std_z = np.std(z, ddof=1)
        mean_z = np.mean(z)
        z_zscore = (z - mean_z) / std_z
        outlier_ind = np.abs(z_zscore) > 10
        z_min = np.min(z[~outlier_ind])
        z_max = np.max(z[~outlier_ind])

        Z = mask.astype('int')
        for idx in range(no_pix):
            # obtain the position in 2D picture 
            h = obj_h[idx]
            w = obj_w[idx]
            Z[h, w] = (z[idx] - z_min) / (z_max - z_min) * 255
        depth = Z
        return depth
    def compute_depth_(Image_name,mask,N):
        """
        compute the depth picture
        """
        im_h, im_w = mask.shape
        N = np.reshape(N, (im_h, im_w, 3))

        # =================get the non-zero index of mask=================
        obj_h, obj_w = np.where(mask != 0)
        no_pix = np.size(obj_h) #37244
        full2obj = np.zeros((im_h, im_w))
        for idx in range(np.size(obj_h)):
            full2obj[obj_h[idx], obj_w[idx]] = idx

        M = scipy.sparse.lil_matrix((2*no_pix, no_pix))
        v = np.zeros((2*no_pix, 1))

        # # ================= fill the M&V =================
        for idx in range(no_pix):
            # obtain the 2D coordinate
            h,p=0,0
            h = obj_h[idx]
            w = obj_w[idx]
            # obtian the surface normal vector
            n_x = N[h, w, 0]
            n_y = N[h, w, 1]
            n_z = N[h, w, 2]
            
            row_idx = idx * 2
            if w + 1 < mask.shape[1] and mask[h, w+1]:
                idx_horiz = int(full2obj[h, w+1])
                M[row_idx, idx] = -1
                M[row_idx, idx_horiz] = 1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_x / n_z
            elif mask[h, w-1]:
                idx_horiz =int( full2obj[h, w-1])
                M[row_idx, idx_horiz] = -1
                M[row_idx, idx] = 1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_x / n_z

            row_idx = idx * 2 + 1
            if h + 1 < mask.shape[0] and mask[h+1, w]:
                idx_vert = int(full2obj[h+1, w])
                M[row_idx, idx] = 1
                M[row_idx, idx_vert] = -1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_y / n_z
            elif mask[h-1, w]:
                idx_vert = int(full2obj[h-1, w])
                M[row_idx, idx_vert] = 1
                M[row_idx, idx] = -1
                if n_z==0:
                    v[row_idx] = 0
                else:
                    v[row_idx] = -n_y / n_z

        # =================sloving the linear equations Mz = v=================
        MtM = M.T @ M
        Mtv = M.T @ v
        z = scipy.sparse.linalg.spsolve(MtM, Mtv)

        std_z = np.std(z, ddof=1)
        mean_z = np.mean(z)
        z_zscore = (z - mean_z) / std_z
        outlier_ind = np.abs(z_zscore) > 10
        z_min = np.min(z[~outlier_ind])
        z_max = np.max(z[~outlier_ind])
        Z=mask.astype('int')
        if(Image_name == 'noisy_venus'):
            Z = mask.astype('float')
        for idx in range(no_pix):
            # obtain the position in 2D picture 
            h = obj_h[idx]
            w = obj_w[idx]
            # Replace NaN values in z with 0
            z = np.nan_to_num(z, nan=0)
            # Now perform the conversion and assignment
            Z[h, w] = (z[idx] - z_min) / (((z_max - z_min) * 255)+0.00001)
        depth = Z
        return depth 
    def Load_light_sources(file_path):
                # Open the file for reading
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Initialize an empty list to store the parsed data
        L = []

        # Loop through each line in the file
        for line in lines:
            # Split the line by ':'
            parts = line.split(':')
            
            # Extract the tuple string and remove any extra characters
            tuple_str = parts[1].strip()[1:-1]
            
            # Split the tuple string by ','
            values = tuple_str.split(',')
            
            # Convert the string values to integers
            values = [int(value.strip()) for value in values]
            
            # Append the tuple to the list
            L.append(tuple(values))
        normalized_light_vectors = L / np.linalg.norm(L, axis=1, keepdims=True)
        L=normalized_light_vectors
        return L
    
    def generate_mask(Image_name):
        # Load the image
        image = cv2.imread("test/"+Image_name+"/pic1.bmp")
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to create a binary image
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create an empty mask
        mask = np.zeros_like(gray)
        if(Image_name == 'noisy_venus'):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # blur
            blur = cv2.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)
            # divide
            divide = cv2.divide(gray, blur, scale=255)
            # otsu threshold
            thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            # apply morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            # cv2.drawContours(morph, contours, -1, (255), thickness=cv2.FILLED)
            cv2.imwrite("test/"+Image_name+"/mask.bmp", morph)
        else:
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
            cv2.imwrite("test/"+Image_name+"/mask.bmp", mask)


    def save_depthmap(depth,filename=None):
        """save the depth map in npy format"""
        if filename is None:
            raise ValueError("filename is None")
        # np.save(filename, depth)
        normalized_arr = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)

        # Convert the array to an image
        image = cv2.applyColorMap(normalized_arr, cv2.COLORMAP_JET)

        # Save the image to a file
        cv2.imwrite("results/filename.jpg", image)

    def disp_depthmap(depth=None, delay=0, name=None):
        """display the depth map"""
        depth = np.uint8(depth)
        if name is None:
            name = 'depth map'
        # cv2.imshow(name, depth)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
    # def process_case(case_folder):
    #     image_files = [f for f in os.listdir(case_folder) if f.endswith('.bmp')]
    #     light_source_file = os.path.join(case_folder, 'LightSource.txt')

    #     # Load light source information
    #     light_source = np.loadtxt(light_source_file)

    #     for image_file in image_files:
    #         # Process image to generate point cloud
    #         image_path = os.path.join(case_folder, image_file)
    #         point_cloud = process_image_to_point_cloud(image_path, light_source)

    #         # Save point cloud as PLY
    #         output_file = os.path.splitext(image_file)[0] + ".ply"
    #         save_ply(point_cloud, output_file)
            
    def parse_light_sources(light_source_file):
        light_sources = {}
        with open(light_source_file) as f:
            for line in f:
                key, value = line.strip().split(': ')
                light_sources[key.strip()] = tuple(map(int, value.strip()[1:-1].split(',')))
        return light_sources
    
    # def load_images(case_folder):
    #     images = []
    #     for i in range(1, 7):  # Assuming images are named pic1.bmp, pic2.bmp, ..., pic6.bmp
    #         img_path = os.path.join(case_folder, f'pic{i}.bmp')
    #         img = cv2.imread(img_path)
    #         if img is not None:
    #             images.append(img)
    #     image_row,image_col= images[0].shape[:2]
    #     return images,image_row,image_col
    
    def create_mask(images):
        mask = np.zeros((image_row,image_col))
        for i in range(image_row):
            for j in range(image_col):
                if images[0][i][j][0]!=0 or images[0][i][j][1]!=0 or images[0][i][j][2]!=0:
                    mask[i][j]=1
        return mask
    
    def Load_masks(self,Image_name):
        mask = cv2.imread('test/'+Image_name+'/mask.bmp')
        self.mask2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        height,width,_=mask.shape
        self.image_row = height
        self.image_col = width
        dst=np.zeros((height,width,3),np.uint8)
        for k in range(3):
            for i in range(height):
                for j in range(width):
                    dst[i,j][k]=255-mask[i,j][k]
        return mask, self.mask2,dst,height,width
    
    def load_images(mask2,Image_name):
        I = []
        if Image_name == 'noisy_venus':
            for i in range(1,7):
                picture = np.array(Image.open('test/'+Image_name+"/"'pic'+str(i)+'.bmp'),'f')
                picture = cv2.cvtColor(picture,cv2.COLOR_RGB2GRAY)
                height, width = picture.shape #(340, 512)
                mask2=np.array(mask2)
                mask2_ = cv2.bitwise_not(mask2)
                picture[mask2_!=0] = 0
                picture = picture.reshape((-1,1)).squeeze(1)
                I.append(picture)
        else:
            for i in range(1,7):
                picture = np.array(Image.open('test/'+Image_name+"/"'pic'+str(i)+'.bmp'),'f')
                picture = cv2.cvtColor(picture,cv2.COLOR_RGB2GRAY)
                height, width = picture.shape 
                picture = picture.reshape((-1,1)).squeeze(1)
                I.append(picture)
        return I,height,width
    
    def RGB2BGR(normal,dst,height,width):
        N = np.reshape(normal.copy(),(height, width, 3))
        # RGB to BGR
        N[:,:,0], N[:,:,2] = N[:,:,2], N[:,:,0].copy()
        N = (N + 1.0) / 2.0
        result = N + dst
        result = result *255
        return result,N
        
    def compute_surfNorm(I, L, mask):
        '''compute the surface normal vector'''
        N = np.linalg.lstsq(L, I, rcond=None)[0].T
        N = normalize(N, axis=1)    
        return N
    
    def show_surfNorm(img,steps=3):
        height,width,_ = img.shape
        dst=np.zeros((height,width,3),np.float64)
        for i in range(3):
            for x in range(0,height,steps):
                for y in range(0,width,steps):
                    dst[x][y][i]=img[x][y][i]
        return dst
    
    ###################### Visualizations ######################
    def normal_visualization(N,image_col,image_row):
        N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
        # Rescale to [0,1] float number
        N_map = (N_map + 1.0) / 2.0
        plt.figure()
        plt.imshow(N_map)
        plt.title('Normal map')
        
    def mask_visualization(M,image_row, image_col):
        mask = np.copy(np.reshape(M, (image_row, image_col)))
        plt.figure()
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        
    def depth_visualization(self,D):
        D_map = np.copy(np.reshape(D, (self.image_row,self.image_col)))
        D = np.uint8(D)
        plt.figure()
        plt.imshow(D_map)
        plt.colorbar(label='Distance to Camera')
        plt.title('Depth map')
        plt.xlabel('X Pixel')
        plt.ylabel('Y Pixel')
        
    def save_ply(self,Z,filepath):
        Z_map = np.reshape(Z, (self.image_row,self.image_col)).copy()
        data = np.zeros((self.image_row*self.image_col,3),dtype=np.float32)
        # let all point float on a base plane 
        baseline_val = np.min(Z_map)
        Z_map[np.where(Z_map == 0)] = baseline_val
        for i in range(self.image_row):
            for j in range(self.image_col):
                idx = i * self.image_col + j
                data[idx][0] = j
                data[idx][1] = i
                data[idx][2] = Z_map[self.image_row - 1 - i][j]
        # output to ply file
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data)
        o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

    # show the result of saved ply file
    def show_ply(filepath):
        pcd = o3d.io.read_point_cloud(filepath)
        o3d.visualization.draw_geometries([pcd])

    # read the .bmp file
    def read_bmp(filepath):
        global image_row
        global image_col
        image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        image_row , image_col = image.shape
        return image
    
    def denoise(image):
        image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        return image
    
    def calibrate_depth(Z,Image_name,mask2):
        if(Image_name == 'bunny'):
            desired_min = -40
            desired_max = 10
        elif(Image_name == 'star'):
            desired_min = -10
            desired_max = 10
        elif(Image_name == 'venus'):
            desired_min = -4
            desired_max = 4
        else:
            desired_min = -10
            desired_max = 10

        # Get the current minimum and maximum values in the depth matrix
        current_min = np.min(Z)
        current_max = np.max(Z)

        # Scale the depth matrix to the desired range
        calibrated_depth_matrix = ((Z - current_min) / (current_max - current_min)) \
                                * (desired_max - desired_min) + desired_min
        # Ensure that the calibrated depth matrix stays within the desired range
        calibrated_depth_matrix = np.clip(calibrated_depth_matrix, desired_min, desired_max)
        mask2 = cv2.bitwise_not(mask2)
        calibrated_depth_matrix[mask2!=0] = 0
        return calibrated_depth_matrix
for Image_name in image_names:
    # Define the file path
    file_path = 'test/'+Image_name+'/LightSource.txt'
    L=PMSR.Load_light_sources(file_path)
    PMSR.generate_mask(Image_name)
    mask,mask2,dst,height,width = PMSR.Load_masks(PMSR,Image_name)
    I,image_col,image_row = PMSR.load_images(mask2,Image_name)
    I = np.array(I)
    # PMSR.mask_visualization(mask2,height,width)
    if(Image_name =='noisy_venus'):
        I= cv2.GaussianBlur(I, (5, 5), 0)
    normal = PMSR.compute_surfNorm(I, L,mask2)
    result,N = PMSR.RGB2BGR(normal,dst,height,width)
    cv2.imwrite("results/normal_files/"+Image_name+".bmp",result)
    result,N= PMSR.RGB2BGR(normal,dst,height,width)
    # PMSR.normal_visualization(normal,width,height)

    if(Image_name=='noisy_venus'):
        Z=PMSR.compute_depth_(Image_name,mask=mask2.copy(),N=normal.copy())
    else: 
        Z = PMSR.compute_depth(mask=mask2.copy(),N=normal.copy())
    # PMSR.save_depthmap(Z,filename="results/est_depth")
    # PMSR.disp_depthmap(depth=Z,name="height")
    calibrated_depth_matrix = PMSR.calibrate_depth(Z,Image_name,mask2)
    # PMSR.depth_visualization(PMSR,calibrated_depth_matrix)
    PMSR.save_ply(PMSR,calibrated_depth_matrix,'results/ply_files/'+Image_name+'.ply')
    

