import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
# def color_thresh(img, rgb_thresh=(160, 160, 160)):
#     # Create an array of zeros same xy size as img, but single channel
#     color_select = np.zeros_like(img[:,:,0])
#     # Require that each pixel be above all three threshold values in RGB
#     # above_thresh will now contain a boolean array with "True"
#     # where threshold was met
#     above_thresh = (img[:,:,0] > rgb_thresh[0]) \
#                 & (img[:,:,1] > rgb_thresh[1]) \
#                 & (img[:,:,2] > rgb_thresh[2])
#     # Index the array of zeros with the boolean array and set to 1
#     color_select[above_thresh] = 1
#     # Return the binary image
#     return color_select


def color_thresh(img, rgb_thresh_low=(160, 160, 160), rgb_thresh_high=(255, 255, 255)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    pix = (img[:,:,0] > rgb_thresh_low[0]) & (img[:,:,0] < rgb_thresh_high[0])\
         & (img[:,:,1] > rgb_thresh_low[1]) & (img[:,:,1] < rgb_thresh_high[1]) \
         & (img[:,:,2] > rgb_thresh_low[2]) & (img[:,:,2] < rgb_thresh_high[2]) 
                
    # Index the array of zeros with the boolean array and set to 1
    color_select[pix] = 1
    # Return the binary image
    return color_select


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

def pix_to_world_float(xpix, ypix, xpos, ypos, yaw, world_size, scale, prec):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran*prec), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran*prec), 0, world_size - 1)
    # Return the result
    return np.float(x_pix_world/prec), np.float(y_pix_world/prec)

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img

    image = Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped = perspect_transform(image, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    navigable = color_thresh(warped, (180,180,180),(255,255,255))
    obstacle  = color_thresh(warped, (0,0,0),(30,30,30))
    rock  = color_thresh(warped, (120,90,0),(230,200,70))

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,0] = navigable * 255
    Rover.vision_image[:,:,1] = rock * 255
    Rover.vision_image[:,:,2] = obstacle * 255
    # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image


    # 5) Convert map image pixel values to rover-centric coords
    navi_xpix, navi_ypix = rover_coords(navigable)
    obs_xpix, obs_ypix = rover_coords(obstacle)
    rock_xpix, rock_ypix = rover_coords(rock)
    
    # 6) Convert rover-centric pixel values to world coordinates
    xpos, ypos = Rover.pos
    yaw =  Rover.yaw
    world_size = Rover.worldmap.shape[0]
    scale = 10
    
    navigable_x_world, navigable_y_world    = pix_to_world(navi_xpix, navi_ypix, xpos, ypos, yaw, world_size, scale)
    obstacle_x_world, obstacle_y_world      = pix_to_world(obs_xpix, obs_ypix, xpos, ypos, yaw, world_size, scale)
    rock_x_world, rock_y_world              = pix_to_world(rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)
    
    #x_pos_world, y_pos_world                = pix_to_world(xpos, ypos, xpos, ypos, yaw, world_size, scale)
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
    Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles

    # new_pix_cnt = 0
    # new_xpix =np.zeros(len(navigable_x_world))
    # new_ypix =np.zeros(len(navigable_x_world))
    #
    # for index, elem in enumerate(navi_xpix):
    #
    #     test_xpix = elem
    #     test_ypix = navi_ypix[index]
    #
    #     test_xworld, test_yworld = pix_to_world_float(test_xpix, test_ypix, xpos, ypos, yaw, world_size, scale, 10)
    #
    #
    #     if(Rover.pos_history[np.int(test_xworld *10),np.int(test_yworld*10)] == 0):
    #         print(test_xworld,",",test_yworld,' is new')
    #         new_xpix[new_pix_cnt] = test_xpix
    #         new_ypix[new_pix_cnt] = test_ypix
    #         nex_pix_cnt = new_pix_cnt + 1
    #         # Additional score with 70% probability for pixel where Rover didn't visit before
    #         if(np.random.rand() < 0.7):
    #             navi_xpix[-1] = test_xpix
    #             navi_ypix[-1] = test_ypix
    #
    # if len(navigable_x_world) >0:
    #     print(len(navigable_x_world))
    #     print(new_pix_cnt)
    #     if (float)(new_pix_cnt)/(len(navigable_x_world)) < 0.2:
    #         print("MOSTLY VISITED BEFORE")
    #         navi_xpix = new_xpix
    #         navi_ypix = new_ypix




    # if (np.sum(rock) > 50):
    #     print("GOLD DIGGER MODE")
    #     Rover.nav_dists, Rover.nav_angles = to_polar_coords(rock_xpix, rock_ypix)
    # else:
    #     Rover.nav_dists, Rover.nav_angles = to_polar_coords(navi_xpix, navi_ypix)

    Rover.nav_dists, Rover.nav_angles = to_polar_coords(navi_xpix, navi_ypix)
    # if (Rover.pos_history[np.int(xpos*10), np.int(ypos*10),0] == 0):
    #     print('Newly Visit')
    # else:
    #     print('----- Visit Before')

    # Rover.pos_history[np.int(xpos*10),np.int(ypos*10),0] = 1;

    return Rover