#include "funcoes.h"


//função para gerar um número aleatório entre [0, 1]
double Rand(unsigned int *seed) {
    return (double)rand_r(seed)/(double)(RAND_MAX);
}

//atribui valores aos spins na rede quadrada
void inicia_malha(unsigned int *seed, int Rede[L][L]){
	int i, j;
	double r;

	for(j=0;j<L;j++)
		for(i=0;i<L;i++){
			r = Rand(seed);
			if(r>0.5)
				Rede[i][j]=1;
			else
				Rede[i][j]=-1;
		}
}

//escolhe uma a posição aleatória de um spin
void escolhe_pos(int *p, unsigned int *seed){
	double r;
	
	r = Rand(seed);
	p[0]=r*L;
	r = Rand(seed);
	p[1]=r*L;
}

//calcula a energia da posição escolhida
int hamiltoniana(int *p, int J, int B, int Rede[L][L]){

	int dir, esq, baixo, cima, E, bordas;
	
	//condições de contorno 2D
	dir=(p[0]+1)%L;
	esq=(p[0]+L-1)%L;
	cima=(p[1]+1)%L;
	baixo=(p[1]+L-1)%L;
	
	bordas = Rede[esq][p[1]]+Rede[dir][p[1]]+Rede[p[0]][cima]+Rede[p[0]][baixo];
	
	E = -J*Rede[p[0]][p[1]]*bordas - B*Rede[p[0]][p[1]];
	
	return E;
}

//decide se o spin da posição flipa ou não (com base na energia ou no número aleatório gerado)
int testa_flip(int *p, int *dE, unsigned int *seed, int J, int B, int k, double T, int Rede[L][L]){
	double r;
	
	*dE=-2*hamiltoniana(p, J, B, Rede);
	r = Rand(seed);
	
	if(*dE < 0 || (r <= exp(-(*dE)/(k*T)))){
		Rede[p[0]][p[1]]*=-1;
		return 1;
	}
	else
		return 0;
}

//termalização
void equilibra(unsigned int *seed, int J, int B, int k, double T, int Rede[L][L]){

	int p[2];
	int i, j, dE=0;
	
	for(i=1;i<=Nskip;i++)
		for(j=1;j<=N;j++){
			escolhe_pos(p, seed);
			testa_flip(p, &dE, seed, J, B, k, T, Rede);
		}
}

//calcula a magnetização da rede quadrada
int magnetizacao_total(int Rede[L][L]){

	int i, j, M=0;
	for(j=L-1;j>=0;j--)
		for(i=0;i<L;i++)
			M+=Rede[i][j];
			
	return M;
}

//calcula a energia total da rede quadrada
int energia_total(int J, int B, int Rede[L][L]){

	int p[2];
	int i, j, E=0;
	
	for (j=L-1;j>=0;j--){
	
		p[1]=j;
		
		for(i=0;i<L;i++){
			p[0]=i;
			E+=hamiltoniana(p, J, B, Rede);
		}
	}
	return E;
}
