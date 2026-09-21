# Data acquisition
The acquisition flow is divided in two separate sections executed in different moments:
  - SCRIPT A -> synthetic data acqusition (generated based on the parameters of the Acconeer A12)
  - SCRIPT B -> offline processing and training+testing the RF
The pipeline will be the following:
        
> Synthetic IQ → Range-Doppler maps → Hand-crafted features → RF classifier   

## SCRIPT A - Synthetic data generation

### 2 Kinematics simulation of the 6 movements
In this section I emulate all the 6 movements: walking, bending, drinking, lying, sitting and falling. What I do, since I don't have real data, is to implement this function which represents the human body as 5 scatterer points: Torso, Head, Right Hand, Left Hand, Right leg, Left leg.

I assign a **RCS (Radar Cross Section)** in order to have a different electromagnetic reflection in every body part. For example the Torso has a much bigger surface and refelects more (RCS = 1.0) with respect to the hand (RCS = 0.15) ().

Then we make a 3D mapping *(x,y,z)* of the 5 points evolving in time *t* for avery action:
- **Walking:** takes step on the *y* axis and a sinusoidal oscillation for the hands and legs;
- **Bending:** the Torso and Head bend down and forward;
- **Drinking:** an asymmetric movement highly located on the upper arts;
- **Lying:** The body is still on the bed/surface with a micro-sinusoidal osciallation of 3 mm at 0.3Hz to simulate breathing;
- **Sitting:** It's a vertical kinematic translation of the whole center of mass of the body (Torso and head). The torso lowers down from an height while standing (about 1.2-1.6m) up to the seating height (0.5-0.6m);

