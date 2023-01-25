from mpi4py import MPI
import numpy as np
import random
import sys
from os import system
import time

def affichage(grille, length1, length2):
    for i in range(0, length1):
        for j in range(0, length2):
            if grille[i,j] == True:
                print('*',end='')
            else:
                print(' ',end='')
        print()


def vaisseau (length1, length2):
    grille = np.zeros((length1, length2),dtype='bool')
    grille[1, 5] = True
    grille[2, 6] = True
    grille[3, 2] = True
    grille[3, 6] = True
    grille[4, 3] = True
    grille[4, 4] = True
    grille[4, 5] = True
    grille[4, 6] = True
    return grille

def vivant_mort(etat, n):
    nouvel_etat = False
    if (etat==False and n==3):
        nouvel_etat = True
    if (etat==True and n<4 and n>1):
        nouvel_etat = True
    return nouvel_etat

def etat_cellule(grille, i, j, length1, length2):
    n = 0
    if (grille[i, (j+1)%length2]):
        n+=1
    if (grille[(i-1+length1)%length1, (j+1)%length2]):
        n+=1
    if (grille[(i+1)%length1, (j+1)%length2]):
        n+=1
    if (grille[i, (j-1+length2)%length2]):
        n+=1
    if (grille[(i-1+length1)%length1, (j-1+length2)%length2]):
        n+=1
    if (grille[(i+1)%length1, (j-1+length2)%length2]):
        n+=1
    if (grille[(i-1+length1)%length1, j]):
        n+=1
    if (grille[(i+1)%length1, j]):
        n+=1
    return vivant_mort(grille[i,j],n)

def verification():
    L = len(sys.argv)
    if rank == root:
        if(L < 7):
            miss = 7-L
            print(f"Il manque {miss} arguments.")
            print("Liste des arguments :")
            print("\t- Length1                  : Int")
            print("\t- Length2                  : Int")
            print("\t- Nombre d'itération       : Int")
            print("\t- Identifiant Root         : Int")
            print("\t- Affichage du jeu         : Bool")
            print("\t- Comparaison Séquentiel   : Bool")
            exit()
    else:
        if(L < 7):
            exit()

def iteration(grille, res,  length1, length2):
    for i in range(1, length1):
        for j in range(0, length2):
            res[i, j] = etat_cellule(grille, i, j, length1, length2)
                     

def send_recv_ghost(grille_locale, rank, length2):
    top_ghost = np.zeros((1, length2), dtype='bool')
    bot_ghost = np.zeros((1, length2), dtype='bool')
    
    if rank%2 == 0:
        comm.Issend(grille_locale[0],  dest=(rank - 1)%size, tag=11)
        comm.Issend(grille_locale[-1], dest=(rank + 1)%size, tag=11)
                
        req = comm.Irecv(bot_ghost, source=((rank + 1)%size), tag=11)
        req.wait()
        req = comm.Irecv(top_ghost, source=((rank - 1)%size), tag=11)
        req.wait()
    
    else:
        req = comm.Irecv(bot_ghost, source=((rank + 1)%size), tag=11)
        req.wait()
        req = comm.Irecv(top_ghost, source=((rank - 1)%size), tag=11)
        req.wait()
        
        comm.Issend(grille_locale[-1], dest=(rank + 1)%size, tag=11)
        comm.Issend(grille_locale[0],  dest=(rank + -1)%size, tag=11)
        
    return np.concatenate((top_ghost, grille_locale, bot_ghost), axis=0)


if __name__ =='__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    length1 = int(sys.argv[1])
    length2 = int(sys.argv[2])
    nb_iter = int(sys.argv[3])
    root    = int(sys.argv[4])
    
    verification()

    af_flag = int(sys.argv[5])
    com_seq = int(sys.argv[6])
    
    
    """
        @ Scatter the grille
    """

    nb_row_local = length1 // size
    reste = length1 % size
    
    grille = None
    count = None
    
    
    if rank == root:
        
        t0 = time.time()
        
        res = np.zeros((length1, length2), dtype='bool')
        grille = vaisseau (length1, length2)
        
        count = np.empty(size, dtype='i')
        for i in range(0, size):
            if i < reste:
                count[i] = (nb_row_local + 1) * length2
            else:
                count[i] = nb_row_local * length2
        
        
    if rank < reste:
        nb_row_local += 1
    
    
    """
        @ Itérations
    """    
    for iter in range(0, nb_iter):
        
        # Actualisation de la taille locale (Si tableau non divisible par ex)
        local_size = nb_row_local * length2
        
        # Création de la grille locale
        grille_locale = np.zeros((nb_row_local, length2), dtype='bool')
        
        # Envoie des parties de la grille | Un vers Tous
        comm.Scatterv([grille, count, MPI.BOOL], grille_locale, root)
        
        """
            @ Récupération des ghosts
        """
        grille_locale_ghosted = send_recv_ghost(grille_locale, rank, length2)
        
        
        """
            @ Calcul du "Futur"
        """
        res = np.zeros((nb_row_local + 2, length2), dtype='bool')
        iteration(grille_locale_ghosted, res, nb_row_local + 2, length2)
        
        # Récupération du bon format de la grille actualisée
        grille_locale = res[1:-1]
            
        
        if rank == root:
            resultat = np.zeros((length1, length2), dtype='bool')
        else:
            resultat = None

        # Récupération de chaque partie de la grille | Tous vers un
        comm.Gatherv([grille_locale, local_size, MPI.BOOL], [resultat, count, MPI.BOOL], root)
        
        
        if rank == root:
            grille = resultat
            if(af_flag):
                affichage(resultat, length1, length2)
                time.sleep(0.2)
                system("clear")
    
    if rank == root:
        tf = time.time()
        T_par = round(tf - t0, 4)
        # Si pas de comparaison avec la partie séquentielle, on affiche les résultats maintenant, sinon, après la partie séquentielle
        if not com_seq: 
            print("-"*25)
            print("Résultat parallélisation:")
            print("-"*25)
            print("\tParamètres :")
            print(f"\t\t- Size                : {length1}*{length2}")
            print(f"\t\t- Nombre de Processus : {size}")
            print(f"\t\t- Temps du programme  : {T_par}", end="\n\n")
                
        
    
    """
        @ Comparaison avec la partie séquentielle
    """
    if com_seq:
        if rank == root:
            t0 = time.time()
            
            res = np.zeros((length1, length2), dtype='bool')
            grille = vaisseau (length1, length2)
            if(af_flag):
                affichage(grille, length1, length2)
            for iter in range(0, nb_iter):
                iteration(grille, res, length1, length2)
                grille, res = res, grille
                
                if(af_flag):
                    affichage(grille, length1, length2)
                    time.sleep(0.5)
                    system("clear")
                
            tf = time.time()
            T = round(tf - t0, 4)
            
            print("-"*25)
            print("Résultat parallélisation:")
            print("-"*25)
            print("\tParamètres :")
            print(f"\t\t- Size                : {length1}*{length2}")
            print(f"\t\t- Nombre de Processus : {size}")
            print(f"\t\t- Temps du programme  : {T_par}", end="\n\n")
            
            print("-"*25)
            print("Résultat séquentiel:")
            print("-"*25)
            print("\tParamètres :")
            print(f"\t\t- Size                : {length1}*{length2}")
            print(f"\t\t- Nombre de Processus : {size}")
            print(f"\t\t- Temps du programme  : {T}", end="\n\n")