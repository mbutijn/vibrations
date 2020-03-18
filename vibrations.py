import numpy as np
import control
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Time vector
T = 10; # s
dt = 0.01; # s
t = np.arange(0,T,dt);
N = len(t); # Number samples
startIndex = 1; # Action starts at index

# Mass spring damper properties
m = 1; # mass, kg
k = 10; # stifness, N/m
c = 1; # damping, sec/m 

# Properties
omega_n = np.sqrt(k/m); # Undamped natural frequency [rad/s]
c_crit = 2*np.sqrt(k*m); # Critical damping
zeta = c/c_crit; # damping ratio
if (zeta < 1):
	omega_d = omega_n*np.sqrt(1-zeta**2) # Damped natural frequency [rad/s]

# Tranfer function
num = np.array([c, k]);
den = np.array([m, c, k]);
sys = control.TransferFunction(num, den);

# https://lpsa.swarthmore.edu/Transient/TransInputs/TransImpulse.html
# http://web.mit.edu/2.151/www/Handouts/FirstSecondOrder.pdf
# https://www.brown.edu/Departments/Engineering/Courses/En4/Notes/vibrations_forced/vibrations_forced.htm

# Print results
print('c_crit =',c_crit);
print('zeta =', zeta);
print('omega_n =',omega_n,'rad/s')
if (zeta < 1):
	print('omega_d =',omega_d,'rad/s')

print('H =',sys)

# Input signal
signalType = int(input('Select input signal:\n1 = Impulse;\n2 = Step;\n3 = Ramp;\n4 = Harmonic;\n5 = Stochastic;\n'));

if signalType == 1: # Impulse input
	u = np.zeros(N);
	diracSize = 1;
	u[startIndex] = diracSize/dt;
	#y_a = diracSize * np.exp(-zeta*omega_n*t)*(2*zeta*omega_n*np.cos(omega_d*t)+((omega_n**2*(1-2*zeta**2))/omega_d)*np.sin(omega_d*t)); #generic decaying oscillatory from https://lpsa.swarthmore.edu/LaplaceZTable/LaplaceZFuncTable.html
	y_a = diracSize * np.exp(-zeta*omega_n*t)*(2*zeta*omega_n*np.cos(omega_d*t)+((omega_d*(1-2*zeta**2))/(1-zeta**2))*np.sin(omega_d*t));
	fileName = 'Impulse';
elif signalType == 2: # Step input
	u = np.zeros(N);
	stepHeight = 1;
	u[startIndex:N] = stepHeight;
	#y_a = stepHeight * (1 + (np.exp(-zeta*omega_n*t)/np.sqrt(1-zeta**2))*(2*zeta*np.sin(omega_d*t) - np.sin(omega_d*t+np.arccos(zeta))));
	y_a = stepHeight * (1 + (np.exp(-zeta*omega_n*t)*omega_n/omega_d)*(2*zeta*np.sin(omega_d*t) - np.sin(omega_d*t+np.arccos(zeta))));
	fileName = 'Step';
elif signalType == 3: # Ramp input
	u = np.zeros(N);
	slope = 1;
	u = slope*(t-dt*startIndex);
	u[0:startIndex] = 0;
	y_a = slope*(t - np.exp(-zeta*omega_n*t)*(1/omega_d)*np.sin(omega_d*t));
	fileName = 'Ramp';
elif signalType == 4: # Harmonic input
	u = np.sin(omega_n*t);
	[[[mag]]], phase, omega = control.freqresp(sys,[omega_n]);
	print('magnitude at omega_n =', mag);
	control.damp(sys, doprint=True);
	A = -mag*(np.exp(-c*t/(2*m))-1); # Calculate amplitude
	#y_a = omega_n**2*((np.sqrt((zeta**2-1)*omega_n**2)*(zeta*omega_n*np.exp(t*(-np.sqrt((zeta**2-1)*omega_n**2)-zeta*omega_n))-zeta*omega_n*np.exp(t*(np.sqrt((zeta**2-1)*omega_n**2)-zeta*omega_n))+np.sqrt((zeta**2-1)*omega_n**2)*np.exp(t*(-np.sqrt((zeta**2-1)*omega_n**2)-zeta*omega_n))+np.sqrt((zeta**2-1)*omega_n**2)*np.exp(t*(np.sqrt((zeta**2-1)*omega_n**2)-zeta*omega_n))))/(4*zeta*(zeta**2-1)*omega_n**4)+(2*zeta*np.sin(t*omega_n)-np.cos(t*omega_n))/(2*zeta*omega_n**2));
	fileName = 'Harmonic';
elif signalType == 5:
	u = np.random.normal(0,1,N); # White noise
	fileName = 'Stochastic';

print('Selected input: ' + fileName);
print('');
print('------------------------------------------------------');

# Simulate with function from library
x0 = 0; # Initial state
T, yout, xout = control.forced_response(sys, t, u, x0);

# Show results
plt.figure(0)
plt.plot(t,u);
plt.plot(t,yout);
plt.xlabel('time, s');
plt.ylabel('distance, m');
plt.title(fileName);

if signalType == 4:
	plt.plot(t,A,dashes=[6, 2]);
	plt.legend(['Input', 'Output', 'Amplitude'])
elif signalType == 5:
	plt.legend(['Input', 'Output'])
else:
	plt.plot(t,y_a);
	plt.legend(['Input', 'Output', 'Analytical'])

# Set up for animation
fig1 = plt.figure(1)
axes = plt.axes(xlim=(-5,5), ylim=(-2,10))
lineU, = axes.plot([-3,3],[0,0]);
lineY, = axes.plot([-2,2,2,-2,-2],[4,4,6,6,4]);
lineSpring, = axes.plot([-1.0,-1.0,-1.5,-0.5,-1.5,-0.5,-1.5,-0.5,-1.5,-1.0,-1.0],[0,0.6,0.8,1.2,1.6,2,2.4,2.8,3.2,3.4,4], color = 'k');
lineDamper1, = axes.plot([1,1],[0, 1.6], color = 'k')
lineDamper2, = axes.plot([0.5, 0.5, 1.5, 1.5],[2.8, 1.6, 1.6, 2.8], color = 'k')
lineDamper3, = axes.plot([1,1],[2.4, 4], color = 'k')

# Move figures
plt.figure(0).canvas.manager.window.wm_geometry("+%d+%d" % (30, 50));
fig1.canvas.manager.window.wm_geometry("+%d+%d" % (700, 50));
	
# Init
def init():
	lineU.set_ydata([]);
	lineY.set_ydata([]);
	lineSpring.set_ydata([]);
	lineDamper1.set_ydata([]);
	lineDamper2.set_ydata([])
	lineDamper3.set_ydata([])
	return lineU, lineY, lineSpring, lineDamper1, lineDamper2, lineDamper3

# Animate
deltaT = 1000*dt; # in seconds
def animate(i):
	lineU.set_ydata([u[i], u[i]]) # update the input signal.
	lineY.set_ydata([yout[i]+4, yout[i]+4, yout[i]+6, yout[i]+6, yout[i]+4]) # update the output signal.
	
	e = yout[i] - u[i] + 4;
	lineSpring.set_ydata([u[i], u[i]+0.15*e, u[i]+0.2*e, u[i]+0.3*e, u[i]+0.4*e, u[i]+0.5*e, u[i]+0.6*e, u[i]+0.7*e, u[i]+0.8*e, u[i]+0.85*e, 4+yout[i]])	
	lineDamper1.set_ydata([u[i], u[i]+0.4*e])
	lineDamper2.set_ydata([u[i]+0.7*e, u[i]+0.4*e,u[i]+0.4*e,u[i]+0.7*e])	
	lineDamper3.set_ydata([u[i]+0.6*e, 4+yout[i]])	
	return lineU, lineY, lineSpring, lineDamper1, lineDamper2, lineDamper3

anim = animation.FuncAnimation(fig1, animate, init_func = init, frames = range(N), interval = deltaT, blit = True, repeat = False)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=1/dt, metadata=dict(artist='Me'), bitrate=1800);

# Save the animation
if input('Save file? (Y/N) ') == 'Y':
	anim.save(fileName + '.mp4', writer = writer)
	print('file saved');
else:
	print('file not saved');

print('Done');
plt.show()
