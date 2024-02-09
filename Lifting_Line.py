import matplotlib.pyplot as plt
import numpy as np
import math
import json
# load json inputs
f = open('Solution.txt', 'w')
fn = 'input.json'
data = json.loads(open(fn).read())
n = data['wing']['nodes_per_semispan']
Type = data['wing']['planform']['type']
section_slope = data['wing']['airfoil_lift_slope']
Alpha = data['condition']['alpha_root[deg]']
washout_type = data['wing']['washout']['distribution']
washout_amount = data['wing']['washout']['amount[deg]']
CLd = data['wing']['washout']['CL_design']
planform_view = data['view']['planform']
washout_distribution = data['view']['washout_distribution']
start = data['wing']['aileron']['begin[z/b]']
end = data['wing']['aileron']['end[z/b]']
flap_frac_0 = data['wing']['aileron']['begin[cf/c]']
flap_frac_f = data['wing']['aileron']['end[cf/c]']
hinge_efficiency = data['wing']['aileron']['hinge_efficiency']
aileron_deflection = data['condition']['aileron_deflection[deg]'] * math.pi / 180
p_bar = data['condition']['pbar']
aileron_distribution = data['view']['aileron_distribution']
CL_hat_distributions = data['view']['CL_hat_distributions']
CL_tilde_distributions = data['view']['CL_tilde_distributions']
# initialize variables
n = 2*n-1
b = 1
thetas = []
lengths = []
chord = []
Lengths0 = []
Chord0 = []
wash = []
cfc = []
chi = []
cos_thetas = []
An = []
# Compute thetas and lengths
for i in range(1, n+1):
    theta = (i-1)*math.pi/(n-1)
    thetas.append(theta)
for i in range(0, n):
    L = -1/2*math.cos(thetas[i])
    lengths.append(L)
# For user input planform
if Type == 'file':
    geom_file = data['wing']['planform']['filename']
    Gp = np.loadtxt(geom_file, skiprows=1)
    N = len(Gp)
    for i in range(0, N):
        L = Gp[i][0]
        Lengths0.append(L)
    for i in range(0, N):
        Cor = Gp[i][1]
        if Cor < 0.001:
            Cor = 0.001
            Chord0.append(Cor)
        else:
            Chord0.append(Cor)
    S = np.trapz(Chord0, Lengths0)
    RA = b**2/(2*S)
    lengths1 = np.dot(Lengths0, -1)[::-1]
    Lengths1 = np.delete(lengths1, [N-1])
    chord1 = Chord0[::-1]
    Chord1 = np.delete(chord1, [N-1])
    Lengths = np.concatenate((Lengths1, Lengths0))
    Chord = np.concatenate((Chord1, Chord0))
    chord = np.interp(lengths, Lengths, Chord)
    Cor_0 = np.interp(start, Lengths, Chord)*(flap_frac_0 - 0.75)
    Cor_f = np.interp(end, Lengths, Chord)*(flap_frac_f - 0.75)
# For tapered and elliptic wings
theta_0 = math.acos(-2 * start)
theta_f = math.acos(-2 * end)
# For tapered planform
if Type == 'tapered':
    RA = data['wing']['planform']['aspect_ratio']
    RT = data['wing']['planform']['taper_ratio']
    for i in range(0, n):
        cor = (2 * b) / (RA * (1 + RT)) * (1 - (1 - RT) * abs(math.cos(thetas[i]))) # eqn (6.28)
        chord.append(cor)
    Cor_0 = (2 * b) / (RA * (1 + RT)) * (1 - (1 - RT) * abs(math.cos(theta_0))) * (flap_frac_0 - 0.75)
    Cor_f = (2 * b) / (RA * (1 + RT)) * (1 - (1 - RT) * abs(math.cos(theta_f))) * (flap_frac_f - 0.75)
# For elliptic planform
elif Type == 'elliptic':
    RA = data['wing']['planform']['aspect_ratio']
    for i in range(0, n):
        cor = (4 * b) / (math.pi * RA) * math.sin(thetas[i])  # eqn (6.27)
        if cor < 0.001:
            cor = 0.001
        chord.append(cor)
    Cor_0 = (4 * b) / (math.pi * RA) * math.sin(theta_0) * (flap_frac_0 - 0.75)
    Cor_f = (4 * b) / (math.pi * RA) * math.sin(theta_f) * (flap_frac_f - 0.75)
# Equation of hinge line
m = (Cor_f-Cor_0)/(end-start)
B = -m*start+Cor_0
for i in range(len(lengths)):
    if start < abs(lengths[i]) < end:
        CfC = (m*abs(lengths[i])+B)/chord[i]+0.75
        cfc.append(CfC)
    else:
        CfC = 0
        cfc.append(CfC)
# Calculate C matrix
C = np.zeros((n, n))
for j in range(1, n + 1):
    C[0][j - 1] = j ** 2
    C[n - 1][j - 1] = (-1) ** (j + 1) * j ** 2
for i in range(1, n - 1):
    for j in range(0, n + 1):
        C[i][j - 1] = ((4 * b) / (section_slope * chord[i]) + j / math.sin(thetas[i])) * math.sin(j * thetas[i])
# Fourier coefficients (an)
c = np.linalg.inv(C)
vec = []
for i in range(0, n):
    Vec = 1
    vec.append(Vec)
an = np.dot(c, vec)
# Washout distribution and bn
if washout_type == 'linear':
    for i in range(0, n):
        w = abs(math.cos(thetas[i]))  # eqn (6.30)
        wash.append(w)
    bn = np.dot(c, wash)
elif washout_type == 'optimum':
    cr = np.interp(0, lengths, chord)
    for i in range(0, n):
        w = 1 - (math.sin(thetas[i])/(chord[i]/cr))  # eqn (6.31)
        wash.append(w)
    bn = np.dot(c, wash)
# Chi distribution
for i in range(len(lengths)):
    if lengths[i] < -end:
        flap_effective = 0
    elif -end < lengths[i] < -start:
        theta_flap = math.acos(2*cfc[i]-1)
        flap_effective = hinge_efficiency * (1 - (theta_flap - math.sin(theta_flap)) / math.pi)
    if -start < lengths[i] < start:
        flap_effective = 0
    elif start < lengths[i] < end:
        theta_flap = math.acos(2 * cfc[i] - 1)
        flap_effective = -1 * hinge_efficiency * (1 - (theta_flap - math.sin(theta_flap)) / math.pi)
    if lengths[i] > end:
        flap_effective = 0
    chi.append(flap_effective)
# Fourier Coefficients (cn)
cn = np.dot(c, chi)
# Fourier Coefficients (dn)
for i in range(len(thetas)):
    cos = math.cos(thetas[i])
    cos_thetas.append(cos)
dn = np.dot(c, cos_thetas)
# write C, C inverse and fourier coefficients (an, bn, cn, dn) to solution.txt file.
with open('Solution.txt', 'w') as f:
    f.write('C Matrix \n\n')
    np.savetxt(f, C)
    f.write('\n''C inverse \n\n')
    np.savetxt(f, c)
    f.write('\n''Fourier coefficients (a_n) \n\n')
    np.savetxt(f, an)
    if not washout_type == 'none':
        f.write('\n''Fourier coefficients (b_n) \n\n')
        np.savetxt(f, bn)
    f.write('\n''Fourier coefficients (c_n) \n\n')
    np.savetxt(f, cn)
    f.write('\n''Fourier coefficients (d_n) \n\n')
    np.savetxt(f, dn)
# Rolling moment due to ailerons
Cl_da = -math.pi*RA/4*cn[1]  # eqn (6.24)
Cl_pbar = -math.pi*RA/4*dn[1]  # eqn (6.25)
if p_bar == 'steady':
    P_bar = -Cl_da / Cl_pbar * aileron_deflection
else:
    P_bar = p_bar
Cl = Cl_da * aileron_deflection + Cl_pbar * P_bar
# Calculate coefficients for tapered and user input planform
if not Type == 'elliptic':
    Kd = []
    for i in range(1, n):
        kd = (i + 1) * an[i] ** 2 / (an[0] ** 2)
        Kd.append(kd)
    kD = sum(Kd)  # eqn (6.16)
    kL = (1 - (1 + math.pi * RA / section_slope) * an[0]) / ((1 + math.pi * RA / section_slope) * an[0])  # eqn (6.21)
    if not washout_type == 'none':
        Ew = bn[0] / an[0]  # eqn (6.22)
        KDL = []
        for i in range(1, n):
            kdl = (i + 1) * an[i] / an[0] * (bn[i] / bn[0] - an[i] / an[0])
            KDL.append(kdl)
        kDL = 2 * bn[0] / an[0] * sum(KDL)  # eqn (6.17)
if not washout_type == 'none':
    KDw = []
    for i in range(1, n):
        kdw = (i + 1) * (bn[i] / bn[0] - an[i] / an[0]) ** 2
        KDw.append(kdw)
    kDw = (bn[0] / an[0]) ** 2 * sum(KDw)
# Calculate coefficients for elliptic planform
if Type == 'elliptic':
    kD = 0
    kL = 0
    kDL = 0
    if washout_type == 'linear':
        Ew = 4/(3*math.pi)
    if washout_type == 'optimum':
        Ew = 0
        kDw = 0
es = 1/(1+kD)
alpah0 = 0*math.pi/180
Slope = section_slope / ((1 + section_slope / (math.pi * RA)) * (1 + kL))
if washout_amount == 'optimum':
    if Type == 'elliptic':
        washout = 0
    else:
        washout = kDL * CLd / (2 * kDw * Slope)  # eqn (6.32)
else:
    washout = washout_amount*math.pi/180
if Alpha == 'CL':
    cl = data['condition']['CL']
    if washout_amount == 'optimum':
        alpha = cl/Slope+Ew*washout
    else:
        alpha = cl / Slope + Ew * washout
else:
    alpha = Alpha*math.pi/180
if washout_type == 'none':
    An = an * alpha + cn * aileron_deflection + dn * P_bar
    CL = math.pi * RA * An[0]
    CDi = CL ** 2 / (math.pi * RA * es)
else:
        An = an * alpha - bn * washout + cn * aileron_deflection + dn * P_bar
        CL = math.pi * RA * An[0]
        CDi = (CL ** 2 * (1 + kD) - kDL * CL * Slope * washout + kDw * (Slope * washout) ** 2) / (math.pi * RA)
# Fourier coefficients (An)
A = []
for i in range(2, n+1):
    a = (2*i-1)*An[i-2]*An[i-1]
    A.append(a)
sumA = sum(A)
Cn = math.pi*RA/4*sumA-math.pi*RA*P_bar/8*(An[0]+An[2])
S = []
for i in range(1, n):
    s = i*An[i-1]**2
    S.append(s)
Ssum = sum(S)
Cdi = math.pi*RA*Ssum-math.pi*RA*P_bar/2*An[1]
# Print relevant information
print('kL = ', float(kL))
print('Lift Slope = ', float(Slope), '\n')
print('kD = ', float(kD))
if not washout_type == 'none':
    print('epsilon_Omega = ', float(Ew))
    print('kDL = ', float(kDL))
    print('k_DOmega = ', float(kDw))
    print('\n''C_l,da =', float(Cl_da))
    print('C_l,pbar =', float(Cl_pbar))
    print('\n''At an angle of attack of', float(alpha * 180 / math.pi), 'degrees,')
    if washout_amount == 'optimum':
        print('with an optimum washout of', float(washout*180/math.pi), 'degrees.''\n')
    else:
        print('with', float(washout*180/math.pi), 'degrees of washout.''\n')
else:
    print('C_l,da =', float(Cl_da))
    print('C_l,pbar =', float(Cl_pbar))
    print('\n''At an angle of attack of', float(alpha*180/math.pi), 'degrees:''\n')
print('CL = ', float(CL))
print('\n''Neglecting Aileron Effects:')
print('CDi = ', float(CDi))
print('\n''Including Aileron Effects:')
print('CDi = ', float(Cdi))
print('Cl =', float(Cl))
print('Cn = ', float(Cn))
print('Pbar_steady = ', float(P_bar))
# Graph planform and washout distribution
if planform_view:
    top = (np.array(chord)/(4*b)).tolist()
    middle = [0, 0]
    side = [lengths[0], lengths[n-1]]
    bottom = (np.array(chord)*-3/(4*b)).tolist()
    plt.rcParams['figure.figsize'] = [10, 2]
    plt.plot(lengths, top, color='black')
    plt.plot(side, middle, color='black')
    plt.plot(lengths, bottom, color='black')
    for i in range(0, n):
        y = [bottom[i], top[i]]
        x = [lengths[i], lengths[i]]
        plt.plot(x, y, color='blue')
    CorL = [Cor_f, Cor_0]
    CorR = [Cor_0, Cor_f]
    LocL = [-end, -start]
    LocR = [start, end]
    Loc1 = [-end, -end]
    Cor1 = [Cor_f, Cor_f/(flap_frac_f-0.75)*-0.75]
    Loc2 = [-start, -start]
    Cor2 = [Cor_0, Cor_0 / (flap_frac_0 - 0.75) * -0.75]
    Loc3 = [start, start]
    Cor3 = [Cor_0, Cor_0 / (flap_frac_0 - 0.75) * -0.75]
    Loc4 = [end, end]
    Cor4 = [Cor_f, Cor_f / (flap_frac_f - 0.75) * -0.75]
    plt.plot(LocL, CorL, color='red')
    plt.plot(LocR, CorR, color='red')
    plt.plot(Loc1, Cor1, color='red')
    plt.plot(Loc2, Cor2, color='red')
    plt.plot(Loc3, Cor3, color='red')
    plt.plot(Loc4, Cor4, color='red')
    plt.title('Planform')
    plt.xlabel('z/b')
    plt.ylabel('c/b')
    plt.show()
if not washout_type == 'none' and washout_distribution:
    plt.rcParams['figure.figsize'] = [6, 4]
    plt.plot(lengths, wash, color='black')
    plt.title('Washout Distribution')
    plt.xlabel('z/b')
    plt.ylabel('omega')
    plt.show()
if aileron_distribution:
    plt.rcParams['figure.figsize'] = [6, 4]
    plt.plot(lengths, chi, color='black')
    plt.title('Aileron Distribution')
    plt.xlabel('z/b')
    plt.ylabel('chi')
    plt.show()
Planform_lift = []
Washout_lift = []
Aileron_lift = []
Roll_lift = []
Total_lift = []
Planform_tilde = []
Washout_tilde = []
Aileron_tilde = []
Roll_tilde = []
Total_tilde = []
for i in range(0, n):
    p_lift = []
    w_lift = []
    a_lift = []
    r_lift = []
    for j in range(0, n):
        P_lift = an[j]*math.sin((j+1)*thetas[i])
        p_lift.append(P_lift)
        W_lift = bn[j] * math.sin((j + 1) * thetas[i])
        w_lift.append(W_lift)
        A_lift = cn[j] * math.sin((j + 1) * thetas[i])
        a_lift.append(A_lift)
        R_lift = dn[j] * math.sin((j + 1) * thetas[i])
        r_lift.append(R_lift)
    planform_lift = 4 * alpha * sum(p_lift)
    Planform_lift.append(planform_lift)
    washout_lift = -4 * washout * sum(w_lift)
    Washout_lift.append(washout_lift)
    aileron_lift = 4 * aileron_deflection * sum(a_lift)
    Aileron_lift.append(aileron_lift)
    roll_lift = 4 * P_bar * sum(r_lift)
    Roll_lift.append(roll_lift)
    total_lift = planform_lift+washout_lift+aileron_lift+roll_lift
    Total_lift.append(total_lift)
    Planform_tilde.append(planform_lift/chord[i])
    Washout_tilde.append(washout_lift / chord[i])
    Aileron_tilde.append(aileron_lift / chord[i])
    Roll_tilde.append(roll_lift / chord[i])
    total_tilde = planform_lift/chord[i]+washout_lift/chord[i]+aileron_lift/chord[i]+roll_lift/chord[i]
    Total_tilde.append(total_tilde)
if CL_hat_distributions:
    plt.plot(lengths, Planform_lift, color='blue', label='Planform')
    plt.plot(lengths, Washout_lift, color='green', label='Washout')
    plt.plot(lengths, Aileron_lift, color='red', label='Aileron')
    plt.plot(lengths, Roll_lift, color='purple', label='Roll')
    plt.plot(lengths, Total_lift, color='black', label='Total')
    plt.title('CL_hat Distributions')
    plt.xlabel('z/b')
    plt.ylabel('CL_hat')
    plt.legend(loc="upper right")
    plt.show()
if CL_tilde_distributions:
    plt.plot(lengths, Planform_tilde, color='blue', label='Planform')
    plt.plot(lengths, Washout_tilde, color='green', label='Washout')
    plt.plot(lengths, Aileron_tilde, color='red', label='Aileron')
    plt.plot(lengths, Roll_tilde, color='purple', label='Roll')
    plt.plot(lengths, Total_tilde, color='black', label='Total')
    plt.title('CL_tilde Distributions')
    plt.xlabel('z/b')
    plt.ylabel('CL_tilde')
    plt.legend(loc="upper right")
    plt.show()

