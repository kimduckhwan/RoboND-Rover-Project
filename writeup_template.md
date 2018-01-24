## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg 

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

To detect navigable area, I modify color_selection function (`color_thresh()`) to have rgb_thresh_low and rgb_thresh_high value to set a range of RGB value.

```
pix = (img[:,:,0] > rgb_thresh_low[0]) & (img[:,:,0] < rgb_thresh_high[0])\
         & (img[:,:,1] > rgb_thresh_low[1]) & (img[:,:,1] < rgb_thresh_high[1]) \
         & (img[:,:,2] > rgb_thresh_low[2]) & (img[:,:,2] < rgb_thresh_high[2]) 
```

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 
In `process_image()`,

1) Define source and destination points
I used `[[14, 140], [301 ,140],[200, 96], [118, 96]]` as source pix from the grid image and destinaton, which is set during the class
```
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
```
2) Apply perspective transform 
```
warped = perspect_transform(img, source, destination)
```

3) Apply color threshold for navigable/rock/obstacle
```
navigable = color_thresh(warped, (160,160,160),(255,255,255)) 
// For me, 160 was good reference, I want narrower navigable path than the class materials
// Not clear how to distinguish 'navigable area' and sky (hope sky is blueish rather than gray)
obstable  = color_thresh(warped, (0,0,0),(10,10,10)) 
// Try to pick only "BLACK"
rock  = color_thresh(warped, (120,90,0),(230,200,70)) 
// From the sample ROCK image, I pick RGB value randomly, and find (120,90,0) at the shaded yellow, and 230,200,70 from the bright yellow
```
4) Convert thresholded image pixel values to rover-centric coords (use `rover_coords()`)
```
navi_xpix, navi_ypix = rover_coords(navigable)
obs_xpix, obs_ypix = rover_coords(obstable)
rock_xpix, rock_ypix = rover_coords(rock)
```    
5) Convert rover-centric pixel values to world coords
5-1) Read data from data using data.count 
```
xpos = np.float(data.xpos[data.count])
ypos = np.float(data.ypos[data.count])
yaw  = np.float(data.yaw[data.count])
#print("current data count: {} --> xpos: {}, ypos: {}, yaw: {}".format(data.count, xpos, ypos, yaw))
world_size = data.worldmap.shape[0]
scale = 10
```
5-2) Convert to world coords (`pix_to_world()`)
```
navigable_x_world, navigable_y_world = pix_to_world(navi_xpix, navi_ypix, xpos, ypos, yaw, world_size, scale)
obstacle_x_world, obstacle_y_world = pix_to_world(obs_xpix, obs_ypix, xpos, ypos, yaw, world_size, scale)
rock_x_world, rock_y_world = pix_to_world(rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)
```
6) Update worldmap (to be displayed on right side of screen)
```
data.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
data.worldmap[rock_y_world, rock_x_world, 1] += 1
data.worldmap[navigable_y_world, navigable_x_world, 2] += 1
```
Rest of part is same as skeleton code of ipynb

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
`perception_step()` is almost identical with `process_image()` except,
1) adding Rovor vision feature: Update Rover.vision_image (this will be displayed on left side of screen) based on the example in the comments
```
Rover.vision_image[:,:,0] = navigable * 255
Rover.vision_image[:,:,1] = rock * 255
Rover.vision_image[:,:,2] = obstacle * 255
```
2) Set Rover.navi_dists and Rover.nav_angles for driving
```
Rover.nav_dists, Rover.nav_angles = to_polar_coords(navi_xpix, navi_ypix)
```

In `decision_step()`
1) I modified `Rover.throttle = np.min([Rover.throttle + 0.05 * Rover.throttle_set,10])` to speed up the Rover but guarantee the throttle is not bigger than 10.0

2) Sometimes, Rover is stuck due to the Rock. Although it has 'clear' navigable view, some part of rover is stuck/blocked by the rock. So I trace the history of Rover's velocity, and if Rover.vel is below 0.2, I increment Rover.slow_cnt. If Rover.slow_cnt hits 300, then Rover became 'backward' mode for 20 iterations (20 decision_step() calls) as below:

```
def decision_step(Rover):
...
                    if (Rover.vel < 0.2):
                        Rover.slow_cnt = Rover.slow_cnt+1
                        if Rover.slow_cnt > 300 :
                            print("ROVER IS STUCK !!!! BACKWARD FOR 20 ITERS")
                            Rover.mode = 'backward'
                            Rover.backward_cnt = 20
                            Rover.slow_cnt = 0
...     
        elif Rover.mode == 'backward':
            Rover.throttle = -5

            Rover.backward_cnt = Rover.backward_cnt -1
            if (Rover.backward_cnt ==0):
                Rover.mode = 'forward'
                Rover.throttle = 0
```


#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

**Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. **

I tried to make Rover try to avoid visiting the same place again to increase map coverage. 

I mark the history of Rover's position. Rover.pos_history is 2000 by 2000 size array. In other words, it has 10 times fine grain coordinate then integer xpos & ypos. For all navigable pixels, I tried to check this pixels are marked as 'visited before' or not. If there are non-visited pixels in current navigable pixels, I add them into navigable pixel array **again**. What I expect is adding them **again** will drag steer angle to the non-visited area rather than mean value of all naviable area. However, it didn't work well.

Since I already have 'navigable' mark and 'obstacle' mark as `Rover.vision_image`, saving 'navigable' and '~obstacle' and '~visited' and if map coverage does not include for a long time, then set target destination and make Rover go there could be the next option I guess.

