<h1>Generalized N-Body Solver</h1>
This application is designed to solve an n-body gravitational system for two or more bodies.
I have designed all of this code around a 
[Simple Example](https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767)  
from Towards Data Science. The example I used lacked any sort of class structure and code to animate 3-body solutions. Additionally I've added features such as error inspection
and generalized center of mass calculations.

**1. Define some galactic body objects** 
<p>Give each body a name, mass, initial position, initial velocity</p>
<pre><code>alpha_centauri_a = NamedBody('Alpha Centauri A', m=1.1, r0=np.array([-0.5, 0, 0]), v0=np.array([0.01, 0.01, 0]))
alpha_centauri_b = NamedBody('Alpha Centauri B', m=0.907, r0=np.array([0.5, 0, 0]), v0=np.array([-0.05, 0, -0.1]))
body_c = NamedBody('Body C', m=1.0, r0=[0, 1, 0], v0=[0, -0.01, 0])
two_bodies = [alpha_centauri_a, alpha_centauri_b]
three_bodies = two_bodies + [body_c]
input_bodies = two_bodies
</code></pre>
 
**2. Initialize number of periods and timesteps**  
&nbsp;*a. for a single system:*
<pre><code>p, step = 50, 0.01  
System = NBodySystem(copy.deepcopy(input_bodies), nd, t_step=step, periods=p)  
</code></pre>
&nbsp;*b. For multiple systems:*
<pre><code>p, timesteps = 50, [0.01, 0.1]  
Systems = [NBodySystem(copy.deepcopy(input_bodies), nd, t_step=step, periods=p)  for step in timesteps]  
</code></pre>

**3. Solve each system**  
The solve loop iterates over each system and applies the following:  
 - execute() 					- runs the ODE solve routine and updates the body objects with solutions
 - compute_center_of_mass() 	- computes the system CoM at each timestep from the solutions
 - compute_relative_positions() - computes the position of each body relative to the system CoM
 - save_solutions( ... ) 		- save the position and velocity data to a csv file. Provide a directory and unique tag for each solution<br/><br/>

Example (Note this loop functions with 1 or more systems):
<pre><code>save_ = False
for i, system in enumerate(Systems):
	system.execute()
	system.compute_center_of_mass()
	system.compute_relative_positions()
	system.save_solutions(save=save_, dir_='results//two-body', tag='-' + str(len(system.bodies)) + '-' + str(p) + '-' + str(int(1/timesteps[i])))
</code></pre>

**4. Extract desired data to plot**
	<pre><code>position_data = [{body.name: list(body.r_sol.values()) for body in S.bodies} for S in Systems]
	velocity_data = [{body.name: list(body.v_sol.values()) for body in S.bodies} for S in Systems]
	pos_relative_data = [{body.name: body.r_com for body in S.bodies} for S in Systems]
	</code></pre>

**5. Visualize Solutions - Examples:**
<p>Several examples are provided to demonstrate various use-cases</p>
(1)	Plot position or velocity of a single system's solution
<p>The example accepts of dict of {name: data} pairs where the data is the solution for the given body name</p>

<pre><code>EXAMPLE_single_plot(position_data[0])</code></pre>

<p align="center"><img src="https://github.com/mbbremner/n-body-solver/blob/master/docs/img/ex1-nbody.png" alt="alt text" width="800" height="550"></p>
(2)	Animate the position or velocity of a single system
<pre><code>animate_solution(data_pos=position_data[0], data_com=None)</code></pre>
[Video of Animated Solution](https://www.youtube.com/watch?v=BxWohnyRdR8&feature=youtu.be)  
(3) Visualize position & velocity data for a System on a single figure
<pre><code>EXAMPLE_pos_vel_plot(position_data[0], velocity_data[0], timesteps[0]))</code></pre>
<p align="center"><img src="https://github.com/mbbremner/n-body-solver/blob/master/docs/img/ex3-nbody.png" alt="alt text" width="800" height="457"></p>
(4) Compare position and velocity visually at two different time incriments
<pre><code>EXAMPLE_compare_timesteps(position_data, velocity_data, timesteps)</code></pre>
<p align="center"><img src="https://github.com/mbbremner/n-body-solver/blob/master/docs/img/ex4-nbody-r.png" alt="alt text" width="800" height="457"></p>
<p align="center"><img src="https://github.com/mbbremner/n-body-solver/blob/master/docs/img/ex4-nbody-v.png" alt="alt text" width="800" height="457"></p>
Don't forget plt.show(), as it's not included in the examples.
