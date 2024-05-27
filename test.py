import numpy as np
import matplotlib.pyplot as plt

# Dype havstrømmer målt fra verdensrommet, lenke: https://ektedata.uib.no/oppgaver/dype-havstrommer-malt-fra-verdensrommet/

# Innlastning av havdata 
Dag = np.loadtxt('satellittmålinger.txt', usecols=0, skiprows=1, delimiter="," , dtype=int)
H_oest = np.loadtxt('satellittmålinger.txt', usecols=1, skiprows=1, delimiter="," , dtype=float)
H_vest = np.loadtxt('satellittmålinger.txt', usecols=2, skiprows=1, delimiter="," , dtype=float)
H_forskjell = H_vest - H_oest

# Plotting av data 
plt.plot(Dag,H_oest)
plt.plot(Dag,H_vest)
plt.plot(Dag,H_forskjell)
plt.xlabel("Dag") 
plt.ylabel("Verdi")
plt.title("Satellittmålinger")
plt.legend(["H_øst", "H_vest","H_forskjell"])
plt.show()

# Lineær regresjon 
L1=np.polyfit(Dag,H_forskjell,1)
Lin1 = L1[0]*Dag + L1[1] # y = ax + b

f=Dag/len(Dag)
H = np.column_stack((np.ones(len(Dag)), Dag, np.sin(2*np.pi*f)))
S = np.dot(np.linalg.inv(np.dot(H.T, H)), np.dot(H.T, H_forskjell)) # S = [a, b, c], S = (H^T*H)^(-1) * H^T * H_forskjell

H_est=S[0]+S[1]*Dag+S[2]*np.sin(2*np.pi*f) # y = a + bx + c*sin(2*pi*x/n)

plt.plot(Dag,H_forskjell)
plt.plot(Dag,Lin1)
plt.plot(Dag,H_est)
plt.legend(["H_forskjell","Lin1","H_est"])
plt.show()

# Innlastning av vanntransportdata
Transport = np.loadtxt('vanntransportdata.txt', usecols=1, skiprows=1, delimiter="," , dtype=float)

# Lineær regresjon
T1=np.polyfit(Dag,Transport,1)
T_Lin1 = T1[0]*Dag + T1[1] # y = ax + b

plt.plot(Dag,Transport)
plt.plot(Dag,T_Lin1)
plt.legend(["Transport","T_Lin1"])
plt.show()