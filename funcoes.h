#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define L 16		//tamanho da rede
#define N L*L		//número de spins
#define Nskip 500000	//número de passos para o sistema atingir o equilíbro
#define MESTRE 0

double Rand(unsigned int *seed);

void inicia_malha(unsigned int *seed, int Rede[L][L]);

void escolhe_pos(int *p, unsigned int *seed);

int hamiltoniana(int *p, int J, int B, int Rede[L][L]);

int testa_flip(int *p, int *dE, unsigned int *seed, int J, int B, int k, double T, int Rede[L][L]);

void equilibra(unsigned int *seed, int J, int B, int k, double T, int Rede[L][L]);

int magnetizacao_total(int Rede[L][L]);

int energia_total(int J, int B, int Rede[L][L]);
