HW1 16825 - vinayakp
==============================================

Results

### Q.1

### Cow 360

![SegmentLocal](Results/mesh_360_hardphongshader.gif "segment")

### Dolly Zoom

![SegmentLocal](Results/dolly.gif "segment")

### Q.2

### Tetrahedron

![SegmentLocal](tetrahedron_360.gif "segment")

### Cube

![SegmentLocal](cube_360.gif "segment")

### Q.3
### Retexturing

![SegmentLocal](Results/color_gradient_cow_360.gif "segment")

### Q.4

### Camera Transformations

![SegmentLocal](Results/textured_cow1.jpg "segment") For this image, I rotated the object (I considered object as a reference for all operations) about z-axis 90 degrees in clockwise direction

![SegmentLocal](Results/textured_cow2.jpg "segment") For this case, I rotated the object about y-axis 90 degrees in clockwise direction. I also pushed the object 3 units in -ve x direction and +ve 3 units in z direction

![SegmentLocal](Results/textured_cow3.jpg "segment") For this view, I pushed the object in positive z axis by a unit of 3

![SegmentLocal](Results/textured_cow4.jpg "segment") For this arrangement, I moved the cow 0.5 units in +ve x direction and 0.5 units in -ve y direction

### Q.5

### 5.1. Rendering Generic 3D Representations

![SegmentLocal](Results/pc1.gif "segment") ![SegmentLocal](Results/pc2.gif "segment") ![SegmentLocal](Results/pc3.gif "segment")

### 5.2. Parametric Functions

### Torus

![SegmentLocal](Results/torus360.gif "segment")

### Square Torus

![SegmentLocal](Results/squaretorus360.gif "segment")

### 5.3. Implicit Surfaces

![SegmentLocal](Results/torus_fxn.gif "segment")

#### Trade-offs between meshes and point-clouds

Point clouds are more memory efficient than meshes as they only contain point coordinates whether meshes have point coordinates as well as the face connectivity information. Points are also more suitable for neural network based shape prediciton as well as while working with changing connectivity. However, point clouds produce low quality renders due to the voids in between the points whereas meshes contain connectivity information thats fills the gaps between the points. Meshes are also very compatible with most 3D simulation based softwares.

![SegmentLocal](Results/double_bubble_fxn.gif "segment")

### 6\. Do Something Fun

![SegmentLocal](Results/morphing_shapes.gif "segment")
