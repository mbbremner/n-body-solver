**1. Define some galactic body objects.Give them a name, mass, initial position, initial velocity**
	<pre><code>alpha_centauri_a = NamedBody('Alpha Centauri A', m=1.1, r0=np.array([-0.5, 0, 0]), v0=np.array([0.01, 0.01, 0]))
	alpha_centauri_b = NamedBody('Alpha Centauri B', m=0.907, r0=np.array([0.5, 0, 0]), v0=np.array([-0.05, 0, -0.1]))
	body_c = NamedBody('Body C', m=1.0, r0=[0, 1, 0], v0=[0, -0.01, 0])
	two_bodies = [alpha_centauri_a, alpha_centauri_b]
	three_bodies = two_bodies + [body_c]
	input_bodies = two_bodies
	</pre></code>
 
**2. Initialize number of periods and timesteps**

 a. for a single system: 
 <pre><code>p, step = 50, 0.01  
 System = NBodySystem(copy.deepcopy(input_bodies), nd, t_step=step, periods=p)  
 </pre></code>
 b. For multiple systems:
 <pre><code>p, timesteps = 50, [0.01, 0.1]  
 Systems = [NBodySystem(copy.deepcopy(input_bodies), nd, t_step=step, periods=p)  for step in timesteps]  
 </pre></code>

**3. Solve each system**  
	The solve loop iterates over each system and applies the following: <br/><br/>
		a. execute() 					- runs the ODE solve routine and updates the body objects with solutions <br/>
		b. compute_center_of_mass() 	- computes the system CoM at each timestep from the solutions <br/>
		c. compute_relative_positions() - computes the position of each body relative to the system CoM <br/>
		d. save_solutions( ... ) 		- save the position and velocity data to a csv file <br/><br/>
	Example (Note this loop functions with 1 or more systems):
	<pre><code>save_ = False
    for i, system in enumerate(Systems):
        system.execute()
        system.compute_center_of_mass()
        system.compute_relative_positions()
        system.save_solutions(save=save_, dir_='results//two-body', tag='-' + str(len(system.bodies)) + '-' + str(p) + '-' + str(int(1/timesteps[i])))
	</pre></code>

**4. Extract desired data to plot**
	<pre><code>position_data = [{body.name: list(body.r_sol.values()) for body in S.bodies} for S in Systems]
	velocity_data = [{body.name: list(body.v_sol.values()) for body in S.bodies} for S in Systems]
	pos_relative_data = [{body.name: body.r_com for body in S.bodies} for S in Systems]
	</pre></code>

**5. Visualize data - Hardcoded examples:** <br/>
	(1)	EXAMPLE_single_plot(position_data[0])  
		- Note: position data is indexed as position_data[0] because there are > 1 systems in this example.  
		- simple 3D plot of either position or velocity data (position used here)  
		- accepts a dictionary of name: data pairs   
		<p align="center"><img src="https://github.com/mbbremner/n-body-solver/blob/master/docs/img/ex1-nbody.png" alt="alt text" width="800" height="550"></p>
	(2)	animate_solution(data_pos=position_data[0], data_com=None) <br/>
		- accepts a dictionary of name: data pairs  
		[Video of Animated Solution](https://www.youtube.com/watch?v=BxWohnyRdR8&feature=youtu.be)  
	(3) EXAMPLE_pos_vel_plot(position_data[0], velocity_data[0], timesteps[0])  
		- A single figure for one N-Body-System with two plots: one for position and one for velocity of each body  
		- accepts a dictionary of name: data pairs
		<p align="center"><img src="https://github.com/mbbremner/n-body-solver/blob/master/docs/img/ex3-nbody.png" alt="alt text" width="800" height="457"></p>
	(4) EXAMPLE_compare_timesteps(position_data, velocity_data, timesteps)  
		- When a system is solved for different timesteps, the solutions may be compared  
		- accepts a dictionary of name: data pairs <br/>
		<p align="center"><img src="https://github.com/mbbremner/n-body-solver/blob/master/docs/img/ex4-nbody-r.png" alt="alt text" width="800" height="457"></p>
		<p align="center"><img src="https://github.com/mbbremner/n-body-solver/blob/master/docs/img/ex4-nbody-v.png" alt="alt text" width="800" height="457"></p>
Don't forget plt.show, as it's not included in the examples.


