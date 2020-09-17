#!/usr/bin/env python
# coding: utf-8

### Sodium Octanoate/Decanoate

# Dec 11, 2019
# ∆G=-∆u(j-1)+g(j^(2/3)-1)+h(j^4-1)-εk-kTln(j!/(j-k)!k!)+∆G_correction



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn import datasets, linear_model
from scipy.special import factorial
import getopt,sys
pd.set_option('use_inf_as_na', True)


# def helpnote(argv):

#     ## Default values for the input arguments are defined here
#     ## It can be printed out with help option -h

#     try:
#         opts, args = getopt.getopt(argv,"h")
#         for opt, arg in opts:
#             if opt == "-h":
#                 Note01 = """
#                 Enthalpogram Generator Based On Cluster Free Energy Profile\n
#                 VERSION : 1.00\n
#                 2019.11.18\n
#                 Xiaokun Zhang\n\n

#                 This program allows for analysis of enthalpy changes in micelle formation process \
#                 and prints out enthalpograms which can be compared to isothermal titration calorimetry(ITC) experiments.\
#                 The design of the program is based on PEACH method developed by Kindt group. \
#                 Please refer to following literature for details of PEACH method ITC experiments.\n\n\n
#                 Two files needed to be prepared for this program. The input note "input.txt" and a file containing\
#                 the equilibrium constants.\n
#                 Input file should be named as "input.txt". \
#                 The following parameters should be included in the input file.\
#                 "input.txt" example:
#                 ECtable.txt
#                 Maibaum1
#                 False
#                 310

#                 Or for user defined model:
#                 ECtable.txt
#                 UserDefined 1 1 1
#                 True
#                 310
#                 2.2


#                 Note:
#                 Line1: File name for cluster free energy profile. This file should include 4 columns, with no header.\n
#                 Delimiters should be comma or tab or whitespace.\n
#                 Surfactant number j, counterion number k, temperature T, equilibrium constant K from PEACH\n\n

#                 Line2: Specify a phenomenological function to fit the cluster free energy. Options are as follows.\n
#                 Maibaum1/Maibaum2/Maibaum3/Quasi_droplet/UserDefined\n

#                 Line3: Temperature
#                 Line4: stock solution concentration

#                 The phenomenological models are derived from classical nucleation theory.\n
#                 For Maibaum models, ∆u term is the bulk free energy of transfering a surfactant from solvent to micelle
#                 ,g term is the surface free energy, h term corresponds structural limitations and is undetermined yet.
#                 ε term accounts for the binding free energy to a micelle, the the factorial term accounts for
#                 the number of ways to distribute k counterions among j surfactant headgroups
#                 ,∆G_correction accounts for other non-ideal effects. \n\n
#                 Maibaum1\n
#                 ∆G=-∆u(j-1)+g(j^(2/3)-1)+h(j^4-1)-εk-kTln(j!/(j-k)!/k!)+∆G_correction\n
#                 Maibaum2\n
#                 ∆G=-∆u(j-1)+g(j^(2/3)-1)+h(j^2-1)-εk-kTln(j!/(j-k)!/k!)+∆G_correction\n
#                 Maibaum3 (Repulsion included)(!!!not included yet)\n
#                 ∆G=-∆u(j-1)+g(j^(2/3)-1)+h(j^2-1)-εk+σ(1-k/j)^2*(j-1)^2-kTln(j!/(j-k)!k!)+∆G_correction\n
#                 Quasi-droplet\n
#                 ∆G=-∆u(j^(3/2)-1)+g(j-1)+h(j^2-1)-εk-kTln(j!/(j-k)!/k!)+∆G_correction\n
#                 UserDefined Model\n
#                 At the last line of the input file, input a b c for a functional form of\
#                 ∆G=-∆u(j^a-1)+g(j^b-1)+h(j^c-1)-εk-kTln(j!/(j-k)!/k!)+∆G_correction\n
#                 """
#                 print(Note01)
#     except:
#         print("run program")


#### Handle input
def Model_parameter(fitmodel):
    if len(fitmodel.split()) > 1:
        ## this is user defined model
        Model, a, b, c = fitmodel.split()
    elif fitmodel == "Maibaum1":
        a,b,c =[1.0,2.0/3.0,4.0]
    elif fitmodel == "Maibaum2":
        a,b,c =[1.0,2.0/3.0,2.0]
    elif fitmodel == "Quasi_droplet":
        a,b,c =[3.0/2.0,1.0,2.0]
    else:
        print("Model Unidentified")
        sys.exit(2)

    return (a,b,c)




def funcH(deltaG,coef_fit,coef_power,T_ref):
    return (coef_fit[1]+coef_fit[2]*(deltaG["T"]-T_ref))*(np.power(deltaG["x"],coef_power[0])-1)+(coef_fit[4]+coef_fit[5]*(deltaG["T"]-T_ref))*(np.power(deltaG["x"],coef_power[1])-1)+(coef_fit[7]+coef_fit[8]*(deltaG["T"]-T_ref))*(np.power(deltaG["x"],coef_power[2])-1)-(coef_fit[10]+coef_fit[11]*(deltaG["T"]-T_ref))*deltaG["y"]

def modelfuncx(deltaG,coef,T_ref):
    A=(deltaG["T"]/T_ref)*(np.power(deltaG["x"],coef)-1)
    A_d=(1-deltaG["T"]/T_ref)*(np.power(deltaG["x"],coef)-1)
    A_dd=-(deltaG["T"]*np.log(deltaG["T"]/T_ref)+T_ref-deltaG["T"])*(np.power(deltaG["x"],coef)-1)
    return A,A_d,A_dd
def modelfuncy(deltaG,T_ref):
    A=-(deltaG["T"]/T_ref)*deltaG["y"]
    A_d=-(1-deltaG["T"]/T_ref)*deltaG["y"]
    A_dd=(deltaG["T"]*np.log(deltaG["T"]/T_ref)+T_ref-deltaG["T"])*deltaG["y"]
    return A,A_d,A_dd

def func_cp(deltaG,coef_fit,coef_power,T_ref):
    return (coef_fit[2]*(np.power(deltaG["x"],coef_power[0])-1)+coef_fit[5]*(np.power(deltaG["x"],coef_power[1])-1)+coef_fit[8]*(np.power(deltaG["x"],coef_power[2])-1)-coef_fit[11]*deltaG["y"])


class LR_fit(object):
    def __init__(self,Ktable,a,b,c,FITTWO,Ttest,T_ref):
        self.Ktable = Ktable
        self.a = a
        self.b = b
        self.c = c
        self.FITTWO=FITTWO
        self.Ttest = Ttest
        self.T_ref = T_ref



    def __call__(self):
        print("RunFIT")
        fitdict={}
        deltaG = self.Ktable.copy()


        deltaG["fac"] = np.log(factorial(deltaG["x"])/factorial(deltaG["x"]-deltaG["y"])/factorial(deltaG["y"]))
        deltaG["G"]= -np.log(deltaG["K"])

        deltaG["G_plusfac"] = deltaG["G"] +deltaG["fac"]
        deltaG.dropna(inplace=True)



        ###Compute coefficients
        coef=[self.a,self.b,self.c]
        deltaG[["A","A_d","A_dd"]]= deltaG.apply(modelfuncx,axis=1,result_type="expand",args=(coef[0],self.T_ref,))
        deltaG[["B","B_d","B_dd"]]= deltaG.apply(modelfuncx,axis=1,result_type="expand",args=(coef[1],self.T_ref,))
        deltaG[["C","C_d","C_dd"]]= deltaG.apply(modelfuncx,axis=1,result_type="expand",args=(coef[2],self.T_ref,))
        deltaG[["D","D_d","D_dd"]]= deltaG.apply(modelfuncy,axis=1,result_type="expand",args=(self.T_ref,))


        #### Linear regression fitting


        print("FITTWO",FITTWO)
        if FITTWO == "True":
            print("true")
            X_1 = deltaG.loc[(deltaG['x'] < 10),'A':'D_dd']
            y_1 = np.array(deltaG.loc[(deltaG['x'] < 10),"G_plusfac"])
            X_2 = deltaG.loc[(deltaG['x'] >= 10),'A':'D_dd']
            y_2 = np.array(deltaG.loc[(deltaG['x'] >= 10),"G_plusfac"])

            reg_1 = linear_model.LinearRegression(fit_intercept=False).fit(X_1, y_1)
            reg_2 = linear_model.LinearRegression(fit_intercept=False).fit(X_2, y_2)

            fitdict["coef1"] = reg_1.coef_
            fitdict["coef2"] = reg_2.coef_
            fitdict["coef"] = (reg_1.coef_,reg_2.coef_)

            print("reg",reg_1,reg_2)
            print("coef_fit_1",fitdict["coef1"])
            print("coef_fit_2",fitdict["coef2"])
            ### Compute R-squared
            deltaG["yhat"] = 0
            yhat_1 = reg_1.predict(X_1)
            deltaG.loc[(deltaG['x']<10),"yhat"] = yhat_1
            yhat_2 = reg_2.predict(X_2)
            deltaG.loc[(deltaG['x']>=10),"yhat"] = yhat_2

            SS_Residual = sum((y_1-yhat_1)**2) + sum((y_2-yhat_2)**2)
            SS_Total = sum((y_2-np.mean(y_2))**2) + sum((y_2-np.mean(y_2))**2)
            r_sqaured = 1 - (float(SS_Residual))/SS_Total

            fitdict["r_squared"]=r_sqaured

            #### Compute deltaH

            deltaG["H"] = 0
            coef_fit_1 = reg_1.coef_
            deltaG.loc[(deltaG['x']<10),"H"] = deltaG.apply(funcH,axis=1,args=(coef_fit_1,coef,self.T_ref,))
            coef_fit_2 = reg_2.coef_
            deltaG.loc[(deltaG['x']>=10),"H"] = deltaG.apply(funcH,axis=1,args=(coef_fit_2,coef,self.T_ref,))

            ### calculate delta_cp

            deltaG.loc[(deltaG['x']<10),"cp"] = deltaG.apply(func_cp,axis=1,args=(coef_fit_1,coef,self.T_ref,))
            deltaG.loc[(deltaG['x']>=10),"cp"] = deltaG.apply(func_cp,axis=1,args=(coef_fit_2,coef,self.T_ref,))

            #print(deltaG)
            #### Output deltaG
            #deltaG["yhat"]=yhat
            deltaG["G_fit"]= deltaG["yhat"]-deltaG["fac"]
            deltaG["K"]= np.exp(-deltaG["G_fit"])
            fitdict["deltaG"]=deltaG[["x","y","T","K","G_plusfac","yhat","G","G_fit","K","H","cp"]]


            for T in self.Ttest:
                #fitdict["H_ij",T] =np.array(deltaG.loc[(deltaG["T"]==T),"H"])
                fitdict["Koutput",T]= deltaG.loc[(deltaG["T"]==T),["x","y","K","H"]]
                self.df=pd.DataFrame({'x':[0],'y':[1.0],'K':[1.0],'H':[0.0]})
                fitdict["Koutput",T]=pd.concat([self.df,fitdict["Koutput",T]])


            print("True FITDone")
            ###Ouput fitdict Keys: "coef","r_sqaured","Koutput","deltaG"
            return fitdict

        else:
            print("false")
            X=deltaG.loc[:,'A':'D_dd']
            y= np.array(deltaG["G_plusfac"])
            reg = linear_model.LinearRegression(fit_intercept=False).fit(X, y)
            fitdict["coef"]=reg.coef_
            ### Compute R-squared
            yhat = reg.predict(X)

            SS_Residual = sum((y-yhat)**2)
            SS_Total = sum((y-np.mean(y))**2)
            r_sqaured = 1 - (float(SS_Residual))/SS_Total

            fitdict["r_squared"]=r_sqaured

            #### Compute deltaH
            coef_fit = reg.coef_

            deltaG["H"] = deltaG.apply(funcH,axis=1,args=(coef_fit,coef,self.T_ref,))

            ### calculate delta_cp
            deltaG["cp"] = deltaG.apply(func_cp,axis=1,args=(coef_fit,coef,self.T_ref,))
            #### Output deltaG
            deltaG["yhat"]=yhat
            deltaG["G_fit"]=yhat-deltaG["fac"]
            deltaG["K"]= np.exp(-deltaG["G_fit"])
            fitdict["deltaG"]=deltaG[["x","y","T","K","G_plusfac","yhat","G","G_fit","K","H","cp"]]


            for T in self.Ttest:
                #fitdict["H_ij",T] =np.array(deltaG.loc[(deltaG["T"]==T),"H"])
                fitdict["Koutput",T]= deltaG.loc[(deltaG["T"]==T),["x","y","K","H"]]
                self.df=pd.DataFrame({'x':[0],'y':[1.0],'K':[1.0],'H':[0.0]})
                fitdict["Koutput",T]=pd.concat([self.df,fitdict["Koutput",T]])


            print("False FITDone")
            ###Ouput fitdict Keys: "coef","r_sqaured","Koutput","deltaG"
            return fitdict







class MicelleStats(object):
    def __init__(self,Molecule,Koutput,Ttest,cal_V=True):
        self.Molecule = Molecule
        self.Koutput = Koutput
        self.Ttest = Ttest
        self.cal_V = cal_V
        #print(self.Koutput)
    def cal_tot(self,Koutput,c_S1,c_Ion1):
        ## calculate tot conc for assigned monomer concentrations
        Koutput["conc"] = Koutput["K"]*(c_S1**Koutput["x"])*(c_Ion1**Koutput["y"])
        conctot_S = (Koutput["conc"]*Koutput["x"]).sum()
        conctot_Ion = (Koutput["conc"]*Koutput["y"]).sum()
        ## calculate micelle s>6
        conctot_M_micelle = (Koutput["conc"][Koutput["x"] >= 7]).sum()
        conctot_S_micelle = conctot_S - (Koutput["conc"][Koutput["x"]<7]*Koutput["x"][Koutput["x"]<7]).sum()
        conctot_Ion_micelle = conctot_Ion - (Koutput["conc"][Koutput["x"]<7]*Koutput["y"][Koutput["x"]<7]).sum()

        return (conctot_S,conctot_Ion,conctot_S_micelle,conctot_Ion_micelle,conctot_M_micelle)

    def __call__(self):

        ### Output dictionary
        Micelledict={}


        ## Find the corresponding set of concentrations for the molecule
        Moleculefile = "INPUT-"+ self.Molecule + ".txt"
        #print(self.Moleculefile)
        Micelledict["conc"] = pd.read_csv(Moleculefile,header=None)

        f = open(Moleculefile,"r")
        lines = f.read()
        conclist = [conc for conc in lines.split('\n') if lines]
        #print(self.conclist)
        del f

        ## Iterate to find the corresponding monomer conc
        c_Stotlist=[]
        c_S1list=[]
        c_Iontotlist=[]
        c_Ion1list=[]

        c_Stot_micelle_list=[]
        c_Iontot_micelle_list=[]
        c_M_micelle_list=[]

        N=0
        for c_Ion1 in conclist:
            N +=1
            tolerance = 0.000001
            val = 10*tolerance
            c_Ion1 = float(c_Ion1)
            c_S1 = c_Ion1

            # not proper design
            c_Stot,c_Iontot,c_Stot_micelle,c_Iontot_micelle,c_M_micelle = self.cal_tot(self.Koutput,c_S1,c_Ion1)
            hibd = 0.0
            lobd = 0.0
            #print("c_S1",self.c_S1)
            while val > tolerance:
                #print(self.val)
                if c_Stot > c_Iontot:
                    hibd = c_S1
                else:
                    lobd = c_S1
                c_S1 = 0.5* (hibd +lobd)
                #print("c_S1_2",self.c_S1)
                c_Stot,c_Iontot, c_Stot_micelle,c_Iontot_micelle,c_M_micelle = self.cal_tot(self.Koutput,c_S1,c_Ion1)
                #print("c_Stot,c_Iontot",self.c_Stot)
                val = (c_Stot-c_Iontot)**2 /(max(c_Stot,c_Iontot))**2
            ###Write output in a dataframe for each conc
            c_Stotlist.append(c_Stot)
            c_Iontotlist.append(c_Iontot)
            c_S1list.append(c_S1)
            c_Ion1list.append(c_Ion1)

            c_Stot_micelle_list.append(c_Stot_micelle)
            c_Iontot_micelle_list.append(c_Iontot_micelle)
            c_M_micelle_list.append(c_M_micelle)
            #print(self.c_Stot,self.c_S1)
            ###Print micelle stats
        #print(self.Micelledict["conc"])
        #print(len(self.c_Stotlist))
        Micelledict["Micellestats"] = Micelledict["conc"].assign(c_Stot=c_Stotlist,
                                           c_Iontot=c_Iontotlist,c_S1=c_S1list,
                                            c_Ion1=c_Ion1list,Mcount=c_M_micelle_list,
                                            Scount=c_Stot_micelle_list,
                                            Ioncount=c_Iontot_micelle_list)


        ## Micelle Stats Calculation
        Micelledict["Micellestats_ave"] = Micelledict["Micellestats"].copy()
        # Calculate Mcount, Scount, Ioncount,
        # Scount/c_Stot,Scount/Mcount,
        # Ioncount/Scount, unitcount
        # Mcount: # of micelles(i>=7)/V ; sum(c(i,j))
        # Scount: # OS in micelles(i>=7)/V ; sum(i*c(i,j))
        # Ioncount: # Na in micelles(i>=7)/ V;
        # percentage
        # Scount/c_Stot: fraction in micelles>=7:
        # Scount/Mcount: mean size
        # Ioncount/masscount: percent neutral
        # unitcount: sum(c(1,j)) from j=0 to max
        Micelledict["Micellestats_ave"]["fraction_of_Micelles"] = Micelledict["Micellestats_ave"]["Scount"]/Micelledict["Micellestats_ave"]["c_Stot"]
        Micelledict["Micellestats_ave"]["mean size"] = Micelledict["Micellestats_ave"]["Scount"]/Micelledict["Micellestats_ave"]["Mcount"]
        Micelledict["Micellestats_ave"]["percent neutral"] = Micelledict["Micellestats_ave"]["Ioncount"]/Micelledict["Micellestats_ave"]["Scount"]




        Micelledict["stock"]=[Micelledict["Micellestats"].loc[N-1,"c_S1"],Micelledict["Micellestats"].loc[N-1,"c_Ion1"],Micelledict["Micellestats"].loc[N-1,"c_Stot"]]

        if self.cal_V == True:
            ##Find StockConc
            StockConc = Micelledict["Micellestats"].loc[N-1,"c_Stot"]
            ## How to find the volume increase of adding stock
            ### Vnew in unit of L or kg
            Micelledict["Micellestats"]["Vnew"] = 0
            Micelledict["Micellestats"].loc[0,"Vnew"] = 0.000996


            ### Vstock in unit of L or kg
            Micelledict["Micellestats"]["Vstock"] = 0
            Micelledict["Micellestats"].loc[0,"Vstock"] = Micelledict["Micellestats"].loc[0,"Vnew"]*(Micelledict["Micellestats"].loc[1,"c_Stot"]-Micelledict["Micellestats"].loc[0,"c_Stot"])/(StockConc-Micelledict["Micellestats"].loc[1,"c_Stot"])
            for i in range(1,N-2):
                #print(i)
                #print(self.Micelledict["Micellestats"].loc[i,"c_Ion1"])
                Micelledict["Micellestats"].loc[i,"Vnew"] = Micelledict["Micellestats"].loc[i-1,"Vnew"]+ Micelledict["Micellestats"].loc[i-1,"Vstock"]
                Micelledict["Micellestats"].loc[i,"Vstock"]=  Micelledict["Micellestats"].loc[i,"Vnew"]*(Micelledict["Micellestats"].loc[i+1,"c_Stot"]-Micelledict["Micellestats"].loc[i,"c_Stot"])/(StockConc-Micelledict["Micellestats"].loc[i+1,"c_Stot"])
            ### Find the stock solution c_S1 and c_Ion1
            Micelledict["stock"]=[Micelledict["Micellestats"].loc[N-1,"c_S1"],Micelledict["Micellestats"].loc[N-1,"c_Ion1"],Micelledict["Micellestats"].loc[N-1,"c_Stot"]]
            Micelledict["Micellestats"].drop(Micelledict["Micellestats"].tail(2).index,inplace=True)
        ## Output Micelledict keys: "conc","Micellestats","stock", "Micellestats_ave"
        return Micelledict

class Enthalpogram(object):
    def __init__(self,Koutput,Micelle_stats,MW,density_a,density_b,density_water,Ttest,stock_S1,stock_Ion1,StockConc):
        self.Koutput = Koutput
        self.MicelleStats = Micelle_stats
        self.MW = MW
        self.density_a = density_a
        self.density_b = density_b
        self.density_water = density_water
        self.Ttest = Ttest
        self.stock_S1 = stock_S1
        self.stock_Ion1 = stock_Ion1
        self.StockConc= StockConc


    def __call__(self):
        ###Output dict
        H_dict={}

        H_stock=[]
        H_conc=[]
        n_surf = self.MicelleStats["Vstock"]*self.StockConc/0.602
        #print(type(self.MicelleStats))

        ### Calculate stock solution H
        ##### Generate a list of <n_ij>
        deltaG_conc = pd.DataFrame()
        deltaG_conc = self.Koutput
        deltaG_conc["n_ij"]= deltaG_conc["K"]*np.power(self.stock_S1,deltaG_conc["x"])*np.power(self.stock_Ion1,deltaG_conc["y"])
        deltaG_conc["sum_H_n"]=deltaG_conc["H"]*deltaG_conc["n_ij"]

        #H_stock.append(H_stock_conc*V_stock/0.602)
        H_stock_conc=deltaG_conc["sum_H_n"].sum()
        #print(H_stock_conc)


        ### Calculate conc series H
        for i in range(0,self.MicelleStats.shape[0]):
            #print(i)
            deltaG_conc = pd.DataFrame()
            V = self.MicelleStats.loc[i,"Vnew"]
            V_stock = self.MicelleStats.loc[i,"Vstock"]

            ##### Generate a list of <n_ij>
            deltaG_conc = self.Koutput
            deltaG_conc["n_ij"]= deltaG_conc["K"]*np.power(self.MicelleStats.loc[i,"c_S1"],deltaG_conc["x"])\
                        *np.power(self.MicelleStats.loc[i,"c_Ion1"],deltaG_conc["y"])*V/0.602
            #if i < 5:
                #print("i=",i)
                #print(deltaG_conc)
            ##### Calculate H_ij * n_ij
            ### Temporarily change H_1,1
            #deltaG_conc.loc[(deltaG_conc["x"] < 5)& (deltaG_conc["y"] < 5),"H"]=0
            deltaG_conc["sum_H_n"]=deltaG_conc["H"]*deltaG_conc["n_ij"]

            H_stock.append(H_stock_conc*V_stock/0.602)
            #print("V_stock",V_stock)
            #print(i,"H_stock",H_stock_conc*V_stock/0.602)

            H_conc.append(deltaG_conc["sum_H_n"].sum())
            #print("H_conc",deltaG_conc["sum_H_n"].sum())

        #print(V_stock)

        ###Generate the enthalpogram
        H_conc_plot = (np.array(H_conc[1:])-np.array(H_conc[:-1])-np.array(H_stock[:-1]))/n_surf[:-1]
        H_conc_plot_1 = (np.array(H_conc[1:])-np.array(H_conc[:-1]))/n_surf[:-1]
        #H_dict["H_output"]= [self.MicelleStats["c_Stot"],H_conc,n_surf]
        #print("H_conc_2",H_conc[-2])
        #print("H_conc_1",H_conc)
        #print("H_stock",H_stock)
        #print("n_surf",n_surf)
        #print("plot",H_conc_plot)


        #print("shape_3",self.MicelleStats)
        self.MicelleStats["c_tot_mol_kg"]=self.MicelleStats["c_Stot"]/0.602/(1+self.MW*self.MicelleStats["c_Stot"]/0.602)
        self.MicelleStats["density"]=self.density_a*np.power(self.MicelleStats["c_tot_mol_kg"],2)+density_b*self.MicelleStats["c_tot_mol_kg"] + density_water
        self.MicelleStats["c_tot"] = self.MicelleStats["c_tot_mol_kg"]*self.MicelleStats["density"]


        k_Na = 1.38064852*10**(-23)*6.02*10**(23)/1000
        ##df1
        H_dict["H_conc"] = pd.DataFrame()
        H_dict["H_conc"]["c_tot"] = self.MicelleStats["c_tot"]
        H_dict["H_conc"]["Hconc"] = H_conc/n_surf*self.Ttest*k_Na
        H_dict["H_conc"]["Hconc_stock"] = H_stock/n_surf*self.Ttest*k_Na
        H_dict["H_conc"]["nsurf"] = n_surf
        H_dict["H_conc"]["V_stock"] = V_stock

        ##df2
        H_dict["enthalpogram"] = pd.DataFrame()
        H_dict["enthalpogram"]["c_tot"] = self.MicelleStats["c_tot"].iloc[1:]
        H_dict["enthalpogram"].reset_index(drop=True,inplace=True)
        #print(self.MicelleStats["c_tot"])
        #print(H_dict["enthalpogram"]["c_tot"] )
        H_dict["enthalpogram"]["H_plot"] = H_conc_plot*self.Ttest*k_Na
        H_dict["enthalpogram"]["H_plot_1"] = H_conc_plot_1*self.Ttest*k_Na
        #print("ctot",H_dict["enthalpogram"]["c_tot"])

        #print("Hplot",H_dict["enthalpogram"]["H_plot"]/self.Ttest/k_Na)
        ## Output H_dict keys: "enthalpogram","H_conc"
        return H_dict






if __name__ == '__main__':

    ### Print help note if needed
    #helpnote(sys.argv[1:])

    ### Read input
    f = open("INPUT-setup.txt","r")
    lines = f.read()
    CFEFILE,FITMODEL,FITTWO,T_ref,Molecule = [ line for line in lines.split('\n') if line]
    del f
    T_ref = float(T_ref)
    #StockConc = float(StockConc)
    print(Molecule)

    ### Find MW and density coefficient for Molecule
    density_water = 0.997043 ##298.15K
    if Molecule == "Octanoate":
        MW = 0.16619 # in kg/mol
        density_a = -0.0033184
        density_b = 0.030028
    elif Molecule == "Decanoate":
        MW = 0.19425
        density_a = -0.037690
        density_b = 0.032614


    # Transform input K table into dataframes
    Ktable=pd.read_csv(CFEFILE, delimiter="\t|,",header=None,skiprows=0,engine='python')
    Ktable.columns=["x","y","T","K"]
    Ttest = np.array(Ktable["T"].unique()).astype(np.float)
    print("Ttest=",Ttest)
    ncountT = len(Ttest)

    # Assign models based on input
    a,b,c = Model_parameter(FITMODEL)

    ### Fit to phenomenological model
    LR_fit = LR_fit(Ktable,a,b,c,FITTWO,Ttest,T_ref)
    LR_fit_dict = LR_fit.__call__()
    Koutput={}
    for T in Ttest:
        Koutput[T] = LR_fit_dict["Koutput",T]

    #### Print LR fitting parameters
    ###Ouput fitdict Keys: "coef","r_sqaured","Koutput","deltaG"

    with open("EA-LR_setup.txt",'w') as f_set:
        ## Use writelines
        f_set.write("coef ={0}\nr_squared ={1}".format(LR_fit_dict["coef"],LR_fit_dict["r_squared"]))
    f_set.close()

    ## Multiple temperature
    for T in Ttest:
        with open("EA-LR_Koutput_{0}K.txt".format(T), 'w') as f_K:
            f_K.write("Temperature={0}\n".format(T))
            Koutput[T].to_csv(f_K,mode='a',sep='\t',index=True)
        f_K.close()
        del f_K

    deltaG_table = LR_fit_dict["deltaG"]
    LR_fit_dict["deltaG"].to_csv("EA-LR_deltaG.txt",sep='\t',index=True)
    del f_set

    ### Plot cp for i
    #Set up plot grid

    fig = plt.figure(figsize=(10,8))
    for T in Ttest:
        plt_list_x = deltaG_table.loc[(deltaG_table["T"]==T),["x"]]

        plt_list_y = deltaG_table.loc[(deltaG_table["T"]==T),["cp"]]
        plt.scatter(plt_list_x,plt_list_y,label = "{0}K".format(T),s=1)
    plt.title("heat capacity in kT per monomer")
    plt.xlabel("Micelle size")
    plt.legend(frameon=False)
    fig.savefig("heat_capacity.jpg")

    ### Plot deltaH per micelle
    #Set up plot grid
    fig = plt.figure(figsize=(20,10))
    fig.subplots_adjust(hspace=.5)
    titles= ["deltaH per micelle in kT","deltaH per monomer in kT"]
    for plt_title, num in zip(titles,range(1,3)):
        ax= fig.add_subplot(1,2,num)

        for T in Ttest:
            plt_list_x = Koutput[T]["x"]
            plt_list_y = [Koutput[T]["H"],Koutput[T]["H"]/Koutput[T]["x"]]
            ax.scatter(plt_list_x,plt_list_y[num-1],label = "{0}K".format(T),s=0.5)
            ax.set_title(plt_title)
        ax.set_xlabel("Micelle size")
        ax.legend(frameon=False)
    fig.savefig("deltaH_per_micelle.jpg")


    ### Generate micelle statistics
    MicelleStats_ave={}
    Micelle_stats={}
    H_dict={}

    print("enthalpogram_Test",Ttest)
    for T in Ttest:
        print("enthalpogram_Test",T)
        #Micellestats = MicelleStats(Molecule,Koutput[T],T)
        MicelleStats_dict = MicelleStats(Molecule,Koutput[T],T).__call__()
        Micelle_stats[T] = MicelleStats_dict["Micellestats"]
        MicelleStats_ave[T] = MicelleStats_dict["Micellestats_ave"]
        stock_S1,stock_Ion1,StockConc = MicelleStats_dict["stock"]

        ## Output Micelledict keys: "conc","Micellestats","stock"
        #### Print micelle stats parameters
        with open("EA-MicelleStats_{0}K.txt".format(T),'w') as f_MS:
            f_MS.write("Stock Solution c_S1={0},c_Ion1={1},c_Stot={2}\n".format(stock_S1,stock_Ion1,StockConc))
            Micelle_stats[T].to_csv(f_MS,mode='a',sep='\t',index=True)
        f_MS.close()
        del f_MS
        with open("EA-MicelleStats_ave_{0}K.txt".format(T),'w') as f_MS_ave:
            f_MS_ave.write("Stock Solution c_S1={0},c_Ion1={1},c_Stot={2}\n".format(stock_S1,stock_Ion1,StockConc))
            MicelleStats_ave[T].to_csv(f_MS_ave,mode='a',sep='\t',index=True)
        f_MS_ave.close()
        del f_MS_ave



        CalEnthalpogram = Enthalpogram(Koutput[T],Micelle_stats[T],MW,density_a,density_b,density_water,T,stock_S1,stock_Ion1,StockConc)
        H_dict[T] = CalEnthalpogram.__call__()
        # ## Output H_dict keys: "enthalpogram","H_conc"
        # ## not print H_conc for now
        # ### Print output and plot data
        print("T",T)
        print(H_dict[T]["enthalpogram"])
        H_dict[T]["enthalpogram"].to_csv("EA-enthalpogram_{0}K.txt".format(T), sep='\t',index=True)
        H_dict[T]["H_conc"].to_csv("EA-H_{0}K.txt".format(T), sep='\t',index=True)

    # c_Stot=c_Stotlist,
    # c_Iontot=c_Iontotlist,c_S1=c_S1list,
    #  c_Ion1=c_Ion1list,Mcount=c_M_micelle_list,
    #   Scount=c_Stot_micelle_list,
    #   Ioncount=c_Iontot_micelle_list)
        #     Micelledict["Micellestats_ave"]["fraction_of_Micelles"]
        # Micelledict["Micellestats_ave"]["mean size"]
        # Micelledict["Micellestats_ave"]["percent neutral"]

    #Set up plot grid
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=.5)
    titles= ["Monomer Conc in mol/L","Free Ion conc in mol/L","Mean Micelle Size","Percentage of Neutralization","Fraction of Micelles(i>6)"]
    for plt_title, num in zip(titles,range(1,6)):
        ax= fig.add_subplot(3,2,num)

        for T in Ttest:
            plt_list_x = MicelleStats_ave[T]["c_Stot"][:-1]
            plt_list_y = [MicelleStats_ave[T]["c_S1"][:-1],MicelleStats_ave[T]["c_Ion1"][:-1],
                        MicelleStats_ave[T]["mean size"][:-1],MicelleStats_ave[T]["percent neutral"][:-1],
                        MicelleStats_ave[T]["fraction_of_Micelles"][:-1]]
            ax.scatter(plt_list_x,plt_list_y[num-1],label = "{0}K".format(T),s=0.5)
            ax.set_title(plt_title)
        ax.set_xlabel("tot Surfactant Conc(mol/L)")
        ax.legend(frameon=False)
    fig.savefig("EA-micelle.jpg")
    del fig

    ### Generate enthalpogram

    # H_dict={}
    # for T in Ttest:
    #     CalEnthalpogram = Enthalpogram(Koutput[T],Micelle_stats[T],MW,density_a,density_b,density_water,T,stock_S1,stock_Ion1,StockConc)
    #     H_dict[T] = CalEnthalpogram.__call__()

    # ## Output H_dict keys: "enthalpogram","H_conc"
    # ## not print H_conc for now
    # ### Print output and plot data
    #     H_dict[T]["enthalpogram"].to_csv("EA_enthalpogram_{0}K.txt".format(T), sep='\t',index=True)

    plt.figure(figsize=(8,6))
    for T in Ttest:
        print("T1",T)
        plt.scatter(H_dict[T]["enthalpogram"]["c_tot"],H_dict[T]["enthalpogram"]["H_plot"],label = "{0}K".format(T))
    plt.legend(frameon=False)
    plt.xlabel("c(mol/L)")
    plt.ylabel("$\Delta$H(kJ/mol)")
    plt.savefig("EA-enthalpogram.jpg")
    #plt.show()
    plt.figure(figsize=(8,6))
    for T in Ttest:
        print("T2",T)
        if T == 295:
            print(H_dict[T]["enthalpogram"]["c_tot"],H_dict[T]["enthalpogram"]["H_plot_1"])
        plt.scatter(H_dict[T]["enthalpogram"]["c_tot"],H_dict[T]["enthalpogram"]["H_plot_1"],label = "{0}K".format(T))
    plt.legend(frameon=False)
    plt.xlabel("c(mol/L)")
    plt.ylabel("$\Delta$H(kJ/mol)")
    plt.savefig("EA-enthalpogram_without_stock.jpg")
    ### Plot H
    plt.figure(figsize=(8,6))
    for T in Ttest:
        print("T3",T)
        if T == 295:
            print(H_dict[T]["H_conc"]["c_tot"],H_dict[T]["H_conc"]["Hconc"])
        plt.scatter(H_dict[T]["H_conc"]["c_tot"],H_dict[T]["H_conc"]["Hconc"],label = "{0}K".format(T))
    plt.xlabel("c(mol/L)")
    plt.ylabel("$\Delta$H(kJ/mol)")
    plt.legend(frameon=False)
    plt.savefig("EA-enthalpogram_H.jpg")

    # plt.figure(figsize=(8,6))
    # for T in Ttest:
    #     plt.scatter(H_dict[T]["H_conc"]["c_tot"],H_dict[T]["H_conc"]["V_stock"],label = "{0}K".format(T))
    # plt.legend(frameon=False)
    # plt.savefig("EA-enthalpogram_Vstock.jpg")


    # plt.figure(figsize=(8,6))
    # for T in Ttest:
    #     plt.scatter(H_dict[T]["H_conc"]["c_tot"],H_dict[T]["H_conc"]["nsurf"],label = "{0}K".format(T))
    # plt.legend(frameon=False)
    # plt.savefig("EA-enthalpogram_nsurf.jpg")


    plt.figure(figsize=(8,6))
    for T in Ttest:
        plt.scatter(H_dict[T]["H_conc"]["c_tot"],H_dict[T]["H_conc"]["Hconc_stock"],label = "{0}K".format(T))
    plt.legend(frameon=False)
    plt.savefig("EA-enthalpogram_Hstock.jpg")

    ###Plot micelle stats from PEACH
    ### Generate micelle statistics
    MicelleStats_ave_Peach={}
    Micelle_stats_Peach={}
    print("start to run peach")
    for T in Ttest:
        cal_V = False
        ### Temporary insert(0,1,1)

        df_1 = pd.DataFrame({'x':[0],'y':[1.0],'K':[1.0]})
        df_2 = Ktable.loc[(Ktable["T"]==T),['x','y','K']]
        df_3 = pd.concat([df_1,df_2])
        MicelleStats_dict_Peach = MicelleStats(Molecule,df_3,T,cal_V).__call__()
        Micelle_stats_Peach[T] = MicelleStats_dict_Peach["Micellestats"]
        MicelleStats_ave_Peach[T] = MicelleStats_dict_Peach["Micellestats_ave"]
        stock_S1,stock_Ion1,StockConc = MicelleStats_dict_Peach["stock"]

        ## Output Micelledict keys: "conc","Micellestats","stock"
        #### Print micelle stats parameters
        with open("PEACH-MicelleStats_{0}K.txt".format(T),'w') as f_MS:
            f_MS.write("Stock Solution c_S1={0},c_Ion1={1},c_Stot={2}\n".format(stock_S1,stock_Ion1,StockConc))
            Micelle_stats_Peach[T].to_csv(f_MS,mode='a',sep='\t',index=True)
        f_MS.close()
        del f_MS
        with open("PEACH-MicelleStats_ave_{0}K.txt".format(T),'w') as f_MS_ave:
            f_MS_ave.write("Stock Solution c_S1={0},c_Ion1={1},c_Stot={2}\n".format(stock_S1,stock_Ion1,StockConc))
            MicelleStats_ave_Peach[T].to_csv(f_MS_ave,mode='a',sep='\t',index=True)
        f_MS_ave.close()
        del f_MS_ave


    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=.5)

    titles= ["Monomer Conc in mol/L","Free Ion conc in mol/L","Mean Micelle Size","Percentage of Neutralization","Fraction of Micelles(i>6)"]
    for plt_title, num in zip(titles,range(1,6)):
        ax= fig.add_subplot(3,2,num)

        for T in Ttest:
            plt_list_x = MicelleStats_ave_Peach[T]["c_Stot"][:-1]
            plt_list_y = [MicelleStats_ave_Peach[T]["c_S1"][:-1],MicelleStats_ave_Peach[T]["c_Ion1"][:-1],
                        MicelleStats_ave_Peach[T]["mean size"][:-1],MicelleStats_ave_Peach[T]["percent neutral"][:-1],
                        MicelleStats_ave_Peach[T]["fraction_of_Micelles"][:-1]]
            ax.scatter(plt_list_x,plt_list_y[num-1],label = "{0}K".format(T),s=0.5)
            ax.set_title(plt_title)
        ax.set_xlabel("tot Surfactant Conc(mol/L)")
        ax.legend(frameon=False)
    fig.savefig("PEACH-micelle.jpg")


    ###To add:  print cp, micelle cutoff defined, two piece fit, compare small cluster, Baysian optimization, cluster analysis code
