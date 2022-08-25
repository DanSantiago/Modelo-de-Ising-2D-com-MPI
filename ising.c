#include "funcoes.h"

int main(int argc, char **argv){

	//define algumas variaveis necessárias para a paralelização 
	int rank, size;
	
	//inicia a zona paralela
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	//define as variáveis necessárias para executar o algoritmo
	FILE *arq;
	int i, j, p, qtdT, count=0, div, pos[2], dE=0, J = 1, k = 1, B = 0, Msteps = 100000000, Rede[L][L];
	double dT = 0.1, l, E=0, E2_media=0, E_total=0, E2_total=0, M=0, M2_media=0, M_media=0, M_total=0, M2_total=0, Mabs_total=0, T = 5, Tmin = 0.5, deltaT;								
	unsigned int seed = 158235;
		
	//inicia a rede aleatóriamente
	inicia_malha(&seed, Rede);	
	
	//calcula a diferença entre T_f e T_i
	deltaT = T - Tmin;
	
	//calcula a quantidade de passos que o loop da temperatura executa
	qtdT = deltaT/dT;
		
	count = 0;

	//checa se a quantidade de passos divide pelo número de cores, se não, cria passos "fantasmas"	
	while((qtdT%size)!=0){
		qtdT += 1;
		count++;
	}
	
	//calcula a quantidade de passos de temperatura que cada core irá executar
	div = qtdT/size;
		
	//variaveis internas de todos cores
	double divEne[div], divMag[div], divCal[div], divSusc[div], divTemp[div];
	double E_media[qtdT], Mabs_media[qtdT], calor_esp[qtdT], susc_mag[qtdT], Temperatura[qtdT];							

	//checa se o processo está sendo executado no core mestre, se sim, gera o vetor temperatura (contém os valores do loop)				
	if(rank == MESTRE){		
		i=0;
		
		for(l=Tmin; l<=T; l+=dT){
			Temperatura[i] = l;
			i++;	
		}			
	}
	
	//divide as fatias do vetor temperatura para os cores
	MPI_Scatter(Temperatura, div, MPI_DOUBLE, divTemp, div, MPI_DOUBLE, MESTRE, MPI_COMM_WORLD);
			
	//loop da temperatura
	for(p=0; p<div; p++) {

		//termalização
		equilibra(&seed, J, B, k, divTemp[p], Rede);
		
		//observáveis com valores no equilíbrio 
		M = magnetizacao_total(Rede);
		E = energia_total(J, B, Rede);
				
		E_total=0;
		E2_total=0;
		M_total=0;
		M2_total=0;
		Mabs_total=0;

		//loop do Monte Carlo
		for(i=1;i<=Msteps;i++){
			//loop de Metropolis
			for(j=1;j<=N;j++){
				escolhe_pos(pos, &seed);
						
				if(testa_flip(pos, &dE, &seed, J, B, k, divTemp[p], Rede)){
					//ajusta os observáveis
					E+=2*dE;
					M+=2*Rede[pos[0]][pos[1]];
				}
			}
					
			//soma dos observavéis
			E_total+=E;
			E2_total+= E*E;
			M_total+=M;
			M2_total+= M*M;
			Mabs_total+=abs(M);
		}			
		
		//média dos observáveis
		divEne[p]=(E_total/(Msteps*N))*0.5;         		  							//<E> - fator 1/2 pela contagem dupla dos pares
		E2_media=(E2_total/(Msteps*N))*0.25;	 		  								//<E²>	- fator 1/4 idem (1/2*1/2)
		M_media=M_total/(Msteps*N);	 		  											//<M>
		M2_media=M2_total/(Msteps*N);	 		  										//<M²>
		divMag[p]=Mabs_total/(Msteps*N);												//<|M|>
		divCal[p] = (E2_media-(divEne[p]*divEne[p]*N))/(k*divTemp[p]*divTemp[p]);     	//C_v = (<E²> - <E>²*N)/(k*T²) | <E>² multiplicado N porque <E>² vai ter um N² no denominador
		divSusc[p] = (M2_media-(M_media*M_media*N))/(k*divTemp[p]); 					//X = (<M²> - <M>²*N)/(k*T) | <M>² multiplicado N porque <M>² vai ter um N² no denominador
	}
	
	//barreira para garantir que todos cores já terminaram seus processos
	MPI_Barrier(MPI_COMM_WORLD);
	
	//envia as fatias dos vetores gerados para o core MESTRE
	MPI_Gather(divEne, div, MPI_DOUBLE, Mabs_media, div, MPI_DOUBLE, MESTRE, MPI_COMM_WORLD);
	MPI_Gather(divMag, div, MPI_DOUBLE, E_media, div, MPI_DOUBLE, MESTRE, MPI_COMM_WORLD);
	MPI_Gather(divCal, div, MPI_DOUBLE, calor_esp, div, MPI_DOUBLE, MESTRE, MPI_COMM_WORLD);
	MPI_Gather(divSusc, div, MPI_DOUBLE, susc_mag, div, MPI_DOUBLE, MESTRE, MPI_COMM_WORLD);

	//checa se o processo esta sendo executado no core mestre, se sim, salva os valores gerados no arquivo	
	if(rank == MESTRE) {
		arq = fopen("dados.dat", "w");						//arquivo para armazenagem de dados
		
		for(i=0; i<=(qtdT-count); i++)
			fprintf(arq, "%.2lf %lf %lf %lf %lf\n", Temperatura[i], Mabs_media[i], E_media[i], calor_esp[i], susc_mag[i]);
			
		fclose(arq);
		
		printf("Fim do programa.\nObserváveis salvos no arquivo 'dados.dat'\nTabelas na ordem: Temperatura | <Magnetização Absoluta> p/ spin | <Energia> p/ spin | Calor Específico p/ spin | Susceptibilidade Magnética p/ spin\n");		
	}
	
	MPI_Finalize();
	
	return 0;
}

