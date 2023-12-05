#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <complex>
#include <cstdio>
#include <tuple>

using namespace std;

// declaring C function/libraries in the C++ code
extern "C"
{
// wrapper of the Fortran Lapack library into C
#include <stdio.h>
#include <omp.h>
#include <complex.h>
#include <lapacke.h>
}

const double pigreco = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;
const double const_electron_charge=1.602176634;
const double const_vacuum_dielectric_constant=8.8541878128;
const double hbar=6.582119569;
const double conversionNmtoeV=6.2415064799632;
typedef complex<double> C;

// Merges two subarrays of array[].
// First subarray is arr[begin..mid]
// Second subarray is arr[mid+1..end]
void merge(double array[], int const left,
		   int const mid, int const right){
	auto const subArrayOne = mid - left + 1;
	auto const subArrayTwo = right - mid;

	// Create temp arrays
	auto *leftArray = new double[subArrayOne],
		 *rightArray = new double[subArrayTwo];

	// Copy data to temp arrays leftArray[]
	// and rightArray[]
	for (auto i = 0; i < subArrayOne; i++)
		leftArray[i] = array[left + i];
	for (auto j = 0; j < subArrayTwo; j++)
		rightArray[j] = array[mid + 1 + j];

	// Initial index of first sub-array
	// Initial index of second sub-array
	auto indexOfSubArrayOne = 0,
		 indexOfSubArrayTwo = 0;

	// Initial index of merged array
	int indexOfMergedArray = left;

	// Merge the temp arrays back into
	// array[left..right]
	while (indexOfSubArrayOne < subArrayOne &&
		   indexOfSubArrayTwo < subArrayTwo){
		if (leftArray[indexOfSubArrayOne] <=
			rightArray[indexOfSubArrayTwo]){
			array[indexOfMergedArray] =
				leftArray[indexOfSubArrayOne];
			indexOfSubArrayOne++;
		}else{
			array[indexOfMergedArray] =
				rightArray[indexOfSubArrayTwo];
			indexOfSubArrayTwo++;
		}
		indexOfMergedArray++;
	}

	// Copy the remaining elements of
	// left[], if there are any
	while (indexOfSubArrayOne < subArrayOne){
		array[indexOfMergedArray] =
			leftArray[indexOfSubArrayOne];
		indexOfSubArrayOne++;
		indexOfMergedArray++;
	}

	// Copy the remaining elements of
	// right[], if there are any
	while (indexOfSubArrayTwo < subArrayTwo){
		array[indexOfMergedArray] =
			rightArray[indexOfSubArrayTwo];
		indexOfSubArrayTwo++;
		indexOfMergedArray++;
	}
}

// begin is for left index and end is
// right index of the sub-array
// of arr to be sorted */
void mergeSort(double array[], int const begin, int const end){
	// Returns recursively
	if (begin >= end)
		return;

	auto mid = begin + (end - begin) / 2;
	mergeSort(array, begin, mid);
	mergeSort(array, mid + 1, end);
	merge(array, begin, mid, end);
}

double function_determinant(double** matrix){
	double determinant = 0;
	determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2]) 
		- matrix[0][1] * (matrix[2][0] * matrix[1][2] - matrix[1][0] * matrix[2][2]) 
		+ matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[2][0] * matrix[1][1]);
	return determinant;
};
double *function_vector_product(double *a, double *b){
	double *c;
	c = new double[3];
	c[0] = a[1] * b[2] - a[2] * b[1]; 
	c[1] = a[2] * b[0] - a[0] * b[2]; 
	c[2] = a[0] * b[1] - a[1] * b[0];
	return c;
};
double function_scalar_product(double *a, double *b){
	double c = 0;
	for (int i = 0; i < 3; i++)
		c = c + a[i] * b[i];
	return c;
};
complex<double> function_scalar_product(complex<double> *a, complex<double> *b){
	complex<double> c = 0;
	for (int i = 0; i < 3; i++)
		c = c + conj(a[i]) * b[i];
	return c;
};
complex<double> function_scalar_product(complex<double> *a, complex<double> *b, int n){
	complex<double> c = 0;
	for (int i = 0; i < n; i++)
		c = c + conj(a[i]) * b[i];
	return c;
};
///the valence and the conduction states pointed by the variable "pair" are are used to build the excitonic state
complex<double>* function_building_excitonic_state(int *pair,complex<double> **kqstates, complex<double> **kstates,int n_valence,int n_conduction,int n_space){
	complex<double> *excitonic_state;
	excitonic_state=new complex<double>[2*n_space];
	for(int i=0;i<n_space;i++){
		excitonic_state[i]=kqstates[pair[0]+n_valence][i];
		excitonic_state[n_space+i]=kstates[pair[1]][i];
	}
	///the excitonic state has the conduction state pair[0] n_space elements and than the valence state pair[1] n_space elements
	///it is an array of length 2*n_space
	return excitonic_state;
};
/// where direct_exchange points the kind of product to consider
complex<double> function_exciton_product(complex<double>* exc1, complex<double>* exc2, int n_space, int direct_exchange){
	complex<double> product1=C(0,0);
	complex<double> product2=C(0,0);
	if(direct_exchange==0)
		for(int i=0;i<n_space;i++){
			product1=product1+conj(exc1[i])*exc2[i];
			product2=product2+conj(exc1[i+n_space])*exc2[i+n_space];
		}
	else
		for(int i=0;i<n_space;i++){
			product1=product1+conj(exc1[i])*exc1[i+n_space];
			product2=product2+conj(exc2[i+n_space])*exc2[i];
		}
	return product1*product2;
};

complex<double>* function_building_dipoles(complex<double> **ks_states_k_point_derivative,complex<double> **ks_states_k_point,double *ks_energies_k_point,int number_of_valence_bands,int number_of_conduction_bands,int number_wannier_functions, double eta, double epsilon){
	complex<double>* dipole=new complex<double>[number_of_conduction_bands*number_of_valence_bands];
	complex<double> conduction_states[number_of_conduction_bands][number_wannier_functions];
	complex<double> valence_states[number_of_valence_bands][number_wannier_functions];
	
	for(int i=0;i<number_of_conduction_bands;i++)
		for(int w=0;w<number_wannier_functions;w++)
			conduction_states[i][w]=ks_states_k_point_derivative[i+number_of_valence_bands][w];
	
	for(int i=0;i<number_of_valence_bands;i++)
		for(int w=0;w<number_wannier_functions;w++)
			valence_states[i][w]=ks_states_k_point[i][w];

	int e=0;
	complex<double> ratio_factor;
	for(int v=0;v<number_of_valence_bands;v++)
		for(int c=0;c<number_of_conduction_bands;c++){
			ratio_factor=(C(ks_energies_k_point[c+number_of_valence_bands]-ks_energies_k_point[v],eta))/C(epsilon,0);
			dipole[e]=(function_scalar_product(conduction_states[c],valence_states[v],number_wannier_functions))/ratio_factor;
			e++;
		}

	return dipole;
};

class Crystal_Lattice{
private:
	int number_atoms;
	double **atoms_coordinates;
	double **bravais_lattice;
	double volume;

public:
	Crystal_Lattice(){
		number_atoms = 0;
		atoms_coordinates = NULL;
		bravais_lattice = NULL;
		volume = 0.0;
	};
	void push_values(ifstream *bravais_lattice_file, ifstream *atoms_coordinates_file);
	void print();
	int pull_number_atoms(){
		return number_atoms;
	}
	double *pull_sitei_coordinates(int sitei){
		return atoms_coordinates[sitei];
	}
	double **pull_bravais_lattice(){
		return bravais_lattice;
	}
	double **pull_atoms_coordinates(){
		return atoms_coordinates;
	}
	double pull_volume(){
		return volume;
	}
};

void Crystal_Lattice:: push_values(ifstream *bravais_lattice_file, ifstream *atoms_coordinates_file){
	if(bravais_lattice_file==NULL)
		cout<<"No Bravais Lattice file!"<<endl;
	if(atoms_coordinates_file==NULL)
		cout<<"No Atoms Coordinateas file!"<<endl;

	bravais_lattice_file->seekg(0);
	atoms_coordinates_file->seekg(0);
	bravais_lattice = new double *[3];
	for (int i = 0; i < 3; i++){
		bravais_lattice[i] = new double[3];
		for (int j = 0; j < 3; j++)
			*bravais_lattice_file >> bravais_lattice[i][j];
	}

	string line;
	while (atoms_coordinates_file->peek() != EOF){
		getline(*atoms_coordinates_file, line);
		number_atoms++;
	}
	atoms_coordinates = new double *[number_atoms];
	atoms_coordinates_file->seekg(0);
	for (int i = 0; i < number_atoms; i++){
		atoms_coordinates[i] = new double[3];
		for (int j = 0; j < 3; j++)
			*atoms_coordinates_file >> atoms_coordinates[i][j];
	}

	volume = function_determinant(bravais_lattice);
};

void Crystal_Lattice:: print(){
	cout << "Bravais Lattice:" << endl;
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++)
			cout << bravais_lattice[i][j] << " ";
		cout << endl;
	}
	cout << "Atoms Coordinates:" << endl;
	for (int i = 0; i < number_atoms; i++){
		for (int j = 0; j < 3; j++)
			cout << atoms_coordinates[i][j] << " ";
		cout << endl;
	}
};

class K_points{
private:
	double spacing;
	int *bz_number_k_points;
	double *shift;
	double ****bz_k_points;
	double **primitive_vectors;
	double **k_points_list;
	int number_k_points_list;
public:
	K_points(){
		spacing = 0;
		primitive_vectors = NULL;
		bz_k_points = NULL;
		shift = new double[3];
		for (int i = 0; i < 3; i++)
			shift[i] = 0.0;
		number_k_points_list=0;
		k_points_list=NULL;
	}
	void push_bz_values(Crystal_Lattice *crystal_lattice, double spacing, double *shift, int add_to_list);
	double *pull_ijkvalues(int i, int j, int k){
		return bz_k_points[i][j][k];
	}
	double pull_spacing(){
		return spacing;
	}
	double *pull_shift(){
		return shift;
	}
	int *pull_bz_number_k_points(){
		return bz_number_k_points;
	}
	double **pull_primitive_vectors(){
		return primitive_vectors;
	}
	void push_list_values(ifstream *k_points_list_file, int number_k_points_list){
		number_k_points_list=number_k_points_list;
		k_points_list=new double*[number_k_points_list];
		k_points_list_file->seekg(0);
		int counting=0;
		while(k_points_list_file->peek()!=EOF){
			if(counting<number_k_points_list){
				k_points_list[counting]=new double[3];
				*k_points_list_file>>k_points_list[counting][0];
				*k_points_list_file>>k_points_list[counting][1];
				*k_points_list_file>>k_points_list[counting][2];
				counting=counting+1;
			}else
				break;
		}
		k_points_list_file->close();
	}
	void print_bz_values();
	double** pull_list_values(){
		return k_points_list;
	}
	int pull_number_kpoint_list(){
		return number_k_points_list;
	}
};

void K_points:: push_bz_values(Crystal_Lattice *crystal_lattice, double spacing, double *shift, int add_to_list){
	primitive_vectors = new double*[3];
	if(crystal_lattice==NULL)
		cout<<"Missing Crystal Lattice in K points grid building!"<<endl;

	spacing = spacing;
	shift = shift;

	primitive_vectors[0] = function_vector_product(crystal_lattice->pull_bravais_lattice()[1],crystal_lattice->pull_bravais_lattice()[2]);
	primitive_vectors[1] = function_vector_product(crystal_lattice->pull_bravais_lattice()[2],crystal_lattice->pull_bravais_lattice()[0]);
	primitive_vectors[2] = function_vector_product(crystal_lattice->pull_bravais_lattice()[0],crystal_lattice->pull_bravais_lattice()[1]);
	double factor=2*pigreco/crystal_lattice->pull_volume();
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++){
			primitive_vectors[i][j]=factor*primitive_vectors[i][j];
		}
	
	bz_number_k_points=new int[3];
	double tmp_variable;
	for(int i=0;i<3;i++){
		tmp_variable=sqrt((function_scalar_product(primitive_vectors[i],primitive_vectors[i])))/spacing;
		bz_number_k_points[i]=int(tmp_variable);
	}
	
	bz_k_points = new double ***[bz_number_k_points[0]];
	for (int i=0;i<bz_number_k_points[0];i++){
		bz_k_points[i]= new double **[bz_number_k_points[1]];
		for (int j = 0; j < bz_number_k_points[1]; j++){
			bz_k_points[i][j] = new double *[bz_number_k_points[2]];
			for (int k = 0; k < bz_number_k_points[2]; k++)
				bz_k_points[i][j][k] = new double[3];
		}
	}

	for (int i = 0; i < bz_number_k_points[0]; i++)
		for (int j = 0; j < bz_number_k_points[1]; j++)
			for (int k = 0; k < bz_number_k_points[2]; k++)
				for (int r = 0; r < 3; r++)
					bz_k_points[i][j][k][r] = ((double)i / bz_number_k_points[0]) * (shift[r] + primitive_vectors[0][r]) 
												+ ((double)j / bz_number_k_points[1]) * (shift[r] + primitive_vectors[1][r]) 
												+ ((double)k / bz_number_k_points[2]) * (shift[r] + primitive_vectors[2][r]);

	if(add_to_list==1){
		number_k_points_list=bz_number_k_points[0]*bz_number_k_points[1]*bz_number_k_points[2];
		k_points_list=new double*[number_k_points_list];
		int counting=0;
		for (int i = 0; i < bz_number_k_points[0]; i++)
			for (int j = 0; j < bz_number_k_points[1]; j++)
				for (int k = 0; k < bz_number_k_points[2]; k++){
					k_points_list[counting]=new double[3];
					for (int r = 0; r < 3; r++)
						k_points_list[counting][r]=bz_k_points[i][j][k][r];
					counting=counting+1;
				}
	}
};

void K_points::print_bz_values(){
	cout<<"K points grid"<<endl;
	for (int i = 0; i < bz_number_k_points[0]; i++){
		cout<<i<<" ";
		for (int j = 0; j < bz_number_k_points[1]; j++){
			cout<<j<<" ";
			for (int k = 0; k < bz_number_k_points[2]; k++){
				cout<<k<<"  (";
				for (int r = 0; r < 3; r++)
					cout<<bz_k_points[i][j][k][r]<<" ";
				cout << " ) ";
			}
			cout<<endl;
		}
		cout << endl;
	}
};

class Hamiltonian_TB{
private:
	int number_wannier_functions;
	int number_primitive_cells;
	double* weights_primitive_cells;
	int** positions_primitive_cells;
	complex<double> ***hamiltonian;
	double fermi_energy;
public:
	Hamiltonian_TB(){
		number_wannier_functions=0;
		number_primitive_cells=0;
		weights_primitive_cells=NULL;
		positions_primitive_cells=NULL;
		hamiltonian=NULL;
		fermi_energy=0;
	}
	/// reading hamiltonian from wannier90 output (spinorial form)
	void push_values(ifstream *wannier90_hr_file, double fermi_energy);
	complex<double> ***pull_hamiltonian(){
		return hamiltonian;
	}
	int push_number_wannier_functions(){
		return number_wannier_functions;
	}
	lapack_complex_double *FFT(double *k_point);
	std:: tuple<double*,complex<double>**> pull_ks_states(double *k_point);
	std:: tuple<double*,complex<double>**> pull_ks_states_subset(double *k_point, int minimum_valence, int maximum_conduction);
	void print_hamiltonian(){
		for(int i=0;i<number_primitive_cells;i++){
			for(int q=0;q<number_wannier_functions;q++)
				for(int s=0;s<number_wannier_functions;s++){
					for(int r=0;r<3;r++)
						cout<<positions_primitive_cells[i][r]<<" ";
					cout<<q<<" "<<s<<" "<<hamiltonian[i][q][s]<<endl;
				}
		}
	}
	void print_ks_states(double *k_point, int minimum_valence, int maximum_conduction);
};

void Hamiltonian_TB:: push_values(ifstream *wannier90_hr_file, double fermi_energy_tmp)
{
	fermi_energy=fermi_energy_tmp;
	int reading_flag=0;
	if(wannier90_hr_file==NULL){
		cout<<"No Wannier90 Hamiltonian file!"<<endl;
		reading_flag=1;
	}
	string history_time;	
	int counting_primitive_cells;
	double number_primitive_cells_tmp;
	int l; int m; int n;
	complex<double> C(0,0);
	counting_primitive_cells=0;

	cout<<"READING HAMILTONIAN..."<<endl;
	while(reading_flag==0){
		wannier90_hr_file->seekg(0);
		getline(*wannier90_hr_file,history_time);
		*wannier90_hr_file>>number_wannier_functions;
		*wannier90_hr_file>>number_primitive_cells;
		weights_primitive_cells=new double[number_primitive_cells];
		positions_primitive_cells=new int*[number_primitive_cells];
		hamiltonian=new complex<double>**[number_primitive_cells];
		for(int i=0;i<number_primitive_cells;i++){
			positions_primitive_cells[i]=new int[3];
			hamiltonian[i]=new complex<double>*[number_wannier_functions];
			for(int j=0;j<number_wannier_functions;j++)
				hamiltonian[i][j]=new complex<double>[number_wannier_functions];
		}
		int counting_positions=0;
		while(counting_positions<number_primitive_cells){
			*wannier90_hr_file>>weights_primitive_cells[counting_positions];
			counting_positions++;
		}

		//cout<<number_primitive_cells<<endl;
		//cout<<number_wannier_functions<<endl;
		//for(int i=0;i<number_primitive_cells;i++)
		//	cout<<weights_primitive_cells[i]<<" ";
		//cout<<endl;

		counting_primitive_cells=1;
		counting_positions=0;
		double real_hopping;
		double imag_hopping;
		complex<double> tmp_variable;
		double trashing_positions[3];
		int total_elements=number_wannier_functions*number_wannier_functions*number_primitive_cells;
		while(counting_positions<total_elements){
			//cout<<double(counting_positions/total_elements*100)<<endl;
			if(counting_positions == number_wannier_functions*number_wannier_functions*(counting_primitive_cells-1)){
				for(int i=0; i<3; i++)
					*wannier90_hr_file>>positions_primitive_cells[counting_primitive_cells-1][i];
				counting_primitive_cells=counting_primitive_cells+1;
			}else
				for(int i=0;i<3;i++)
					*wannier90_hr_file>>trashing_positions[i];

			*wannier90_hr_file>>l;
			*wannier90_hr_file>>m;
			*wannier90_hr_file>>real_hopping;
			*wannier90_hr_file>>imag_hopping;
			number_primitive_cells_tmp=weights_primitive_cells[counting_primitive_cells-2];
			tmp_variable=real_hopping*number_primitive_cells_tmp+_Complex_I*number_primitive_cells_tmp;
			hamiltonian[counting_primitive_cells-2][l-1][m-1]=tmp_variable;
			//cout<<l<<" "<<m<<" "<<hamiltonian[counting_primitive_cells-2][l-1][m-1]<<endl;
			counting_positions++;
		}
		reading_flag=1;
	}
};

lapack_complex_double *Hamiltonian_TB:: FFT(double *k_point){	

	lapack_complex_double *fft_hamiltonian;
	fft_hamiltonian = (lapack_complex_double *)malloc(number_wannier_functions * number_wannier_functions * sizeof(lapack_complex_double));

	double product_tmp; double cos_tmp; double sin_tmp;
	double real_part; double imag_part;
	for(int l=0;l<number_wannier_functions;l++)
		for(int m=0;m<number_wannier_functions;m++){
			fft_hamiltonian[l*number_wannier_functions+m]=0.0+_Complex_I*(0.0);
			for(int r=0;r<number_primitive_cells;r++){
				product_tmp=0;
				for(int s=0;s<3;s++)
					product_tmp=product_tmp+k_point[s]*positions_primitive_cells[r][s];
				///angle in radians
				cos_tmp=cos(product_tmp);
				sin_tmp=sin(product_tmp);
				real_part=hamiltonian[r][l][m].real()*cos_tmp-hamiltonian[r][l][m].imag()*sin_tmp;
				imag_part=hamiltonian[r][l][m].imag()*cos_tmp+hamiltonian[r][l][m].real()*sin_tmp;
				real_part=real_part+lapack_complex_double_real(fft_hamiltonian[l*number_wannier_functions+m]);
				imag_part=imag_part+lapack_complex_double_imag(fft_hamiltonian[l*number_wannier_functions+m]);
				fft_hamiltonian[l*number_wannier_functions+m]=real_part+_Complex_I*(imag_part);
			}
		}
	return fft_hamiltonian;
};

std:: tuple<double*,complex<double>**> Hamiltonian_TB:: pull_ks_states(double *k_point){
	lapack_complex_double *fft_hamiltonian;
	fft_hamiltonian=FFT(k_point);
	
	///flags for the diagonalization routine
	int N=number_wannier_functions; int LDA=number_wannier_functions; 
	int matrix_layout = 101; int INFO;
	char JOBZ='V';	char UPLO='L';
	///ks_eigenvalues are the eigenvalues
	double* ks_eigenvalues;
	ks_eigenvalues = (double*)malloc(number_wannier_functions*sizeof(double));
	
	INFO=LAPACKE_zheev(matrix_layout, JOBZ, UPLO, N, fft_hamiltonian, LDA, ks_eigenvalues);
	cout<<"INFO "<<INFO<<endl;
	///TEST PRINT
	//for(int i=0;i<number_wannier_functions;i++)
	//	cout<<ks_eigenvalues[i]<<endl;
	//	for(int j=0;j<number_wannier_functions;j++)
	//		cout<<ks_eigenvectors[i][j]<<" ";
	//	cout<<endl;
	//}
	///saving the eigenvalues in order to find after the ordering
	///the correspondence old - new eigenvalues
	double ks_eigenvalues_tmp_with_ordering[number_wannier_functions][2];
	for (int i = 0;i<number_wannier_functions; i++){
		ks_eigenvalues_tmp_with_ordering[i][1]=0;
		ks_eigenvalues_tmp_with_ordering[i][0]=ks_eigenvalues[i];
	}
	///ordering eigenvalues
	mergeSort(ks_eigenvalues, 0, number_wannier_functions - 1);
	///saving the changes of the ordering
	int flag; int j; int new_ordering[number_wannier_functions];
	for (int i=0;i<number_wannier_functions; i++){
		j=0; flag=0;
		while (flag!=1){
			if ((ks_eigenvalues_tmp_with_ordering[j][0] == ks_eigenvalues[i]) && (ks_eigenvalues_tmp_with_ordering[j][1] != 1)){
				new_ordering[i]=j;
				ks_eigenvalues_tmp_with_ordering[j][1]=1;
				flag=1;
			}
			else
				j++;
		}
	}

	///ks_wavefunctions are the eigenvectors
	///here thery are ordered in order to correspond to the ordered eigenvalues 
	complex<double> **ks_eigenvectors;
	ks_eigenvectors=new complex<double>*[number_wannier_functions];
	for(int i=0;i<number_wannier_functions;i++){
		ks_eigenvectors[i]=new complex<double>[number_wannier_functions];
		for(int j=0;j<number_wannier_functions;j++)
			ks_eigenvectors[i][j]=fft_hamiltonian[j*number_wannier_functions+new_ordering[i]];
	}
	free(fft_hamiltonian);

	///normalizing eigenvectors
	complex<double> modulus;
	for(int i=0;i<number_wannier_functions;i++){
		modulus=sqrt(function_scalar_product(ks_eigenvectors[i],ks_eigenvectors[i],number_wannier_functions));
		for(int j=0;j<number_wannier_functions;j++)
			ks_eigenvectors[i][j]=ks_eigenvectors[i][j]/modulus;
	}

	return {ks_eigenvalues,ks_eigenvectors};
};

std:: tuple<double*,complex<double>**> Hamiltonian_TB:: pull_ks_states_subset(double *k_point, int minimum_valence, int maximum_conduction){
	complex<double> **ks_eigenvectors;
	double* ks_eigenvalues;
	int number_valence_bands=0;
	int number_conduction_bands=0;
	std:: tuple<double*,complex<double>**> ks_states;
	
	ks_states=pull_ks_states(k_point);
	ks_eigenvectors=std::get<1>(ks_states);
	ks_eigenvalues=std::get<0>(ks_states);

	for(int i=0;i<number_wannier_functions;i++){
		cout<<ks_eigenvalues[i]<<endl;
		if(ks_eigenvalues[i]<=fermi_energy)
			number_valence_bands++;
		else
			number_conduction_bands++;
	}
	cout<<"Total bands "<<number_valence_bands<<" "<<number_conduction_bands<<"  Fermi energy "<<fermi_energy<<endl;

	if(number_conduction_bands<maximum_conduction)
		cout<<"Too many conduction bands required "<<maximum_conduction<<" "<<number_conduction_bands<<endl;
	
	if(number_valence_bands<minimum_valence)
		cout<<"Too many valence bands required "<<minimum_valence<<number_valence_bands<<endl;
	
	complex<double> **ks_eigenvectors_subset;
	double *ks_eigenvalues_subset;
	int dimensions_subspace=minimum_valence+maximum_conduction;
	ks_eigenvalues_subset=new double[dimensions_subspace];
	ks_eigenvectors_subset=new complex<double>*[dimensions_subspace];
	for(int i=0;i<dimensions_subspace;i++){
		ks_eigenvectors_subset[i]=new complex<double>[number_wannier_functions];
		if(i<minimum_valence){
			ks_eigenvectors_subset[i]=ks_eigenvectors[(number_valence_bands-1)-i];
			ks_eigenvalues_subset[i]=ks_eigenvalues[(number_valence_bands-1)-i];
		}else{
			ks_eigenvectors_subset[i]=ks_eigenvectors[(number_valence_bands-1)+(i-minimum_valence)+1];	
			ks_eigenvalues_subset[i]=ks_eigenvalues[(number_valence_bands-1)+(i-minimum_valence)+1];
		}
	}
	for(int i=0;i<number_wannier_functions;i++)
		delete[] ks_eigenvectors[i];
	delete[] ks_eigenvalues;
	delete[] ks_eigenvectors;
	///first are written valence states, than at higher rows conduction states
	return {ks_eigenvalues_subset,ks_eigenvectors_subset};
};

void Hamiltonian_TB::print_ks_states(double* k_point, int minimum_valence, int maximum_conduction){
	std::tuple<double*,complex<double>**> results_htb;
	double *eigenvalues;
	complex<double> **eigenvectors;
	std::tuple<double*,complex<double>**> results_htb_subset;
	double *eigenvalues_subset;
	complex<double> **eigenvectors_subset;
	results_htb=pull_ks_states(k_point);
	eigenvalues=std::get<0>(results_htb);
	eigenvectors=std::get<1>(results_htb);
	cout<<"ALL BANDS"<<endl;
	for(int i=0;i<number_wannier_functions;i++){
		printf("%d	%.2f\n",i,eigenvalues[i]);
		for(int j=0;j<number_wannier_functions;j++)
			printf("(%.2f,%.2f)",eigenvectors[i][j].real(),eigenvectors[i][j].imag());
		cout<<endl;
	}
	results_htb_subset=pull_ks_states_subset(k_point,minimum_valence,maximum_conduction);
	eigenvalues_subset=std::get<0>(results_htb_subset);
	eigenvectors_subset=std::get<1>(results_htb_subset);
	cout<<"ONLY SUBSET"<<endl;
	for(int i=0;i<minimum_valence+maximum_conduction;i++){
		printf("%d	%.2f\n",i,eigenvalues_subset[i]);
		for(int j=0;j<number_wannier_functions;j++)
			printf("(%.2f,%.2f)",eigenvectors_subset[i][j].real(),eigenvectors_subset[i][j].imag());
		cout<<endl;
	}	
};

class Coulomb_Potential_3D{
private:
	double volume_unit_cell;
	double electron_charge;
	double effective_dielectric_constant;
	double vacuum_dielectric_constant;
	double minimum_k_point_modulus;
public:
	Coulomb_Potential_3D(){
		volume_unit_cell=0;
		electron_charge=const_electron_charge;
		vacuum_dielectric_constant=const_vacuum_dielectric_constant;
		effective_dielectric_constant=1.0;
		minimum_k_point_modulus=0;
	}
	Coulomb_Potential_3D(Crystal_Lattice crystal_lattice,double effective_dielectric_constant_tmp,double minimum_k_point_modulus_tmp){
		volume_unit_cell=crystal_lattice.pull_volume();
		effective_dielectric_constant=effective_dielectric_constant_tmp;
		minimum_k_point_modulus=minimum_k_point_modulus_tmp;
		electron_charge=const_electron_charge;
		vacuum_dielectric_constant=const_vacuum_dielectric_constant;
	}
	double pull(double* k_point);
	double pull_volume_unit_cell(){
		return volume_unit_cell;
	}
};

double Coulomb_Potential_3D:: pull(double* k_point){
	///the volume of the cell is in angstrom
	///the momentum k is in angstrom^-1
	double coulomb_potential;
	double modulus_k_point=sqrt(function_scalar_product(k_point,k_point));
	if(modulus_k_point<minimum_k_point_modulus)
		coulomb_potential=0;
	else
		coulomb_potential=-conversionNmtoeV*100*pow(electron_charge,2)/(2*volume_unit_cell*vacuum_dielectric_constant*effective_dielectric_constant*pow(modulus_k_point,2));
	return coulomb_potential;
};

class Excitonic_Hamiltonian{
private:
	int number_wannier_functions;
	int number_of_valence_bands;
	int number_of_conduction_bands;
	int dimension_bse_hamiltonian_cv;
	int dimension_bse_hamiltonian;
	int number_k_points_list;
	double **k_points_list;
	ifstream *wannier90_hr_file;
	int** exciton;
	Coulomb_Potential_3D *coulomb_potential;
	double fermi_energy;
	double volume_unit_cell;
public:
	Excitonic_Hamiltonian(){
		number_wannier_functions=0;
		number_of_conduction_bands=0;
		number_of_valence_bands=0;
		dimension_bse_hamiltonian_cv=0;
		dimension_bse_hamiltonian=0;
		wannier90_hr_file=NULL;
		fermi_energy=0;
		number_k_points_list=0;
		k_points_list=NULL;
		coulomb_potential=NULL;
		exciton=NULL;
	}
	/// be carefull: do not try to build the BSE matrix with more bands than those given by the hamiltonian!!!
	/// there is a check at the TB hamiltonian level but not here... (it would require to study the Brillouin zone, in any case it can be added easily)
	void push_values(int number_of_valence_bands_tmp,int number_of_conduction_bands_tmp,
		double **k_points_list_tmp,int number_k_points_list_tmp,ifstream *wannier90_hr_file_tmp,double fermi_energy_tmp,
		Crystal_Lattice crystal_lattice_tmp,double effective_dielectric_constant_tmp,double minimum_k_point_modulus_tmp,int number_wannier_functions_tmp){
		number_k_points_list=number_k_points_list_tmp;
		number_of_conduction_bands=number_of_conduction_bands_tmp;
		number_of_valence_bands=number_of_valence_bands_tmp;
		number_wannier_functions=number_wannier_functions_tmp;
		dimension_bse_hamiltonian_cv=number_of_conduction_bands*number_of_valence_bands;
		dimension_bse_hamiltonian=number_k_points_list*number_of_conduction_bands*number_of_valence_bands;
		wannier90_hr_file=wannier90_hr_file_tmp;
		fermi_energy=fermi_energy_tmp;
		k_points_list=k_points_list_tmp;
		coulomb_potential=new Coulomb_Potential_3D(crystal_lattice_tmp,effective_dielectric_constant_tmp,minimum_k_point_modulus_tmp);
		/// defining the excitonic basis
		/// correspondence excitonic state <-> pair conduction valence
		exciton=new int*[dimension_bse_hamiltonian_cv];
		for(int e=0;e<dimension_bse_hamiltonian_cv;e++)
			exciton[e]=new int[2];
		int e=0;
		for(int v=0;v<number_of_valence_bands;v++)
			for(int c=0;c<number_of_conduction_bands;c++){
				exciton[e][0]=c;
				exciton[e][1]=v;
				e++;
			}
	}
	std::tuple<complex<double>**,lapack_complex_double*> pull_hamiltonian_and_renormalized_dipoles(double* excitonic_momentum);
	std::tuple<complex<double>*,complex<double>**> pull_eigenstates_cholesky_way(lapack_complex_double* excitonic_hamiltonian);
	std::tuple<complex<double>*,complex<double>**> pull_eigenstates_usual_way(lapack_complex_double* excitonic_hamiltonian);
	complex<double> ***pull_excitonic_oscillator_force(complex<double> **excitonic_eigenstates, complex<double>* excitonic_eigenergies, complex<double> **dipoles);
	void dielectric_tensor(double *excitonic_momentum, double eta, double* omega, int n, ofstream* file_diel,double scissor_operator);
	void print(double* excitonic_momentum){
		
		std::tuple<complex<double>**,lapack_complex_double*> dipoles_and_hamiltonian;
		std::tuple<complex<double>*,complex<double>**> eigenvalues_and_eigenstates;
		dipoles_and_hamiltonian=pull_hamiltonian_and_renormalized_dipoles(excitonic_momentum);
		complex<double>** dipoles=std::get<0>(dipoles_and_hamiltonian);
		lapack_complex_double *hamiltonian=std::get<1>(dipoles_and_hamiltonian);


		cout<<"BSE HAMILTONIAN"<<endl;
		cout<<"Real Part"<<endl;
		double tmp_variable1;
		double tmp_variable2;
		for (int i=0;i<number_k_points_list;i++){
			for (int q=0;q<dimension_bse_hamiltonian_cv;q++){
				for (int j=0;j<number_k_points_list;j++){
					for (int s=0;s<dimension_bse_hamiltonian_cv;s++){
						tmp_variable1=lapack_complex_double_real(hamiltonian[s+dimension_bse_hamiltonian_cv*j+dimension_bse_hamiltonian*q+dimension_bse_hamiltonian*dimension_bse_hamiltonian_cv*i]);
						printf("%12.2f|", tmp_variable1);
					}
				}
				cout<<endl;
			}
		}
		cout<<endl;
		cout<<"Immaginary Part"<<endl;
		for (int i=0;i<number_k_points_list;i++){
			for (int q=0;q<dimension_bse_hamiltonian_cv;q++){
				for (int j=0;j<number_k_points_list;j++){
					for (int s=0;s<dimension_bse_hamiltonian_cv;s++){
						tmp_variable2=lapack_complex_double_imag(hamiltonian[s+dimension_bse_hamiltonian_cv*j+dimension_bse_hamiltonian*q+dimension_bse_hamiltonian*dimension_bse_hamiltonian_cv*i]);
						printf("%12.2f|", tmp_variable2);
					}
				}
				cout<<endl;
			}
		}

		cout<<"DIPOLES"<<endl;
		for (int i=0;i<3;i++){
			cout<<i<<endl;
			for (int q=0;q<dimension_bse_hamiltonian;q++)
				printf("(%.4f+i%.4f)| ",dipoles[i][q].real(),dipoles[i][q].imag());
			cout<<endl;
		}

		eigenvalues_and_eigenstates=pull_eigenstates_usual_way(hamiltonian);
		complex<double>* eigenvalues=std::get<0>(eigenvalues_and_eigenstates);
		complex<double>** eigenstates=std::get<1>(eigenvalues_and_eigenstates);
		cout<<"DIAGONALISATION"<<endl;
		for (int i=0;i<dimension_bse_hamiltonian;i++)
			cout<<eigenvalues[i]<<endl;		
	}
};

std:: tuple<complex<double>**,lapack_complex_double*> Excitonic_Hamiltonian::pull_hamiltonian_and_renormalized_dipoles(double* excitonic_momentum){
	/// saving the excitonic states over the Brillouin zone, in order to speed up the BSE matrix building
	/// saving dipoles as well to speed up the BSE oscillator force calculation
	///dipoles are already defined with the energy denominator
	complex<double>** dipoles;
	complex<double>** starting_dipoles;
	dipoles=new complex<double>*[3];
	starting_dipoles=dipoles;

	for(int r=0;r<3;r++)
		dipoles[r]=new complex<double>[dimension_bse_hamiltonian];

	double epsilon=0.0001;
	double eta=0.001;
	complex<double> **ks_states_k_point_derivative;
	double *ks_energies_k_point_derivative;

	Hamiltonian_TB hamiltonian_TB;
	hamiltonian_TB.push_values(wannier90_hr_file,fermi_energy);
	cout<<"FERMI ENERGY "<<fermi_energy<<endl;
	double *k_point; double *k_point_q;
	k_point=new double[3]; k_point_q=new double[3];

	complex<double> ***excitonic_states;
	complex<double> **excitonic_energies;
	complex<double> **ks_states_k_point;
	complex<double> **ks_states_k_point_q;
	double *ks_energies_k_point;
	double *ks_energies_k_point_q;
	std:: tuple<double*, complex<double>**> ks_states;

	excitonic_states=new complex<double>**[number_k_points_list];
	excitonic_energies=new complex<double>*[number_k_points_list];
	
	int counting_states=0;
	//#pragma omp parallel for
	for (int i=0;i<number_k_points_list;i++){
		cout<<"status k "<<i<<endl;
		excitonic_states[i]=new complex<double>*[dimension_bse_hamiltonian_cv];
		excitonic_energies[i]=new complex<double>[dimension_bse_hamiltonian_cv];
		for(int r=0;r<3;r++){
			k_point[r]=k_points_list[i][r];
			k_point_q[r]=k_point[r]+excitonic_momentum[r];
			cout<<k_point[r]<<" "<<k_point_q[r]<<" ";
		}
		cout<<endl;
		ks_states=hamiltonian_TB.pull_ks_states_subset(k_point,number_of_valence_bands,number_of_conduction_bands);
		ks_energies_k_point=std::get<0>(ks_states);
		ks_states_k_point=std::get<1>(ks_states);
		ks_states=hamiltonian_TB.pull_ks_states_subset(k_point_q,number_of_valence_bands,number_of_conduction_bands);
		ks_energies_k_point_q=std::get<0>(ks_states);
		ks_states_k_point_q=std::get<1>(ks_states);
		for(int j=0;j<3;j++){
			cout<<"derivative k "<<j<<endl;
			k_point[j]=k_point[j]+epsilon;
			ks_states=hamiltonian_TB.pull_ks_states_subset(k_point,number_of_valence_bands,number_of_conduction_bands);
			ks_energies_k_point_derivative=std::get<0>(ks_states);
			ks_states_k_point_derivative=std::get<1>(ks_states);
			dipoles[j]=function_building_dipoles(ks_states_k_point_derivative,ks_states_k_point,ks_energies_k_point,number_of_valence_bands,number_of_conduction_bands,number_wannier_functions,eta,epsilon);
			dipoles[j]=dipoles[j]+dimension_bse_hamiltonian_cv;
		}
		for(int q=0;q<dimension_bse_hamiltonian_cv;q++){
			excitonic_states[i][q]=function_building_excitonic_state(exciton[q],ks_states_k_point_q,ks_states_k_point,number_of_valence_bands,number_of_conduction_bands,number_wannier_functions);
			//for(int z=0;z<number_wannier_functions;z++)
			//	cout<<excitonic_states[i][q][z]<<" ";
			//cout<<endl;
			excitonic_energies[i][q]=ks_energies_k_point_q[exciton[q][0]]-ks_energies_k_point[exciton[q][1]];
		}
	}

	dipoles=starting_dipoles;

	///since the Coulomb potential varies slightly inside unit cell in comparison with Bloch functions, it has been taken out of the braket evaluation
	
	///// saving memory for the BSE matrix (kernel)
	lapack_complex_double* excitonic_hamiltonian;
	excitonic_hamiltonian=(lapack_complex_double*)malloc(dimension_bse_hamiltonian*dimension_bse_hamiltonian*sizeof(lapack_complex_double));

	///// building the BSE matrix
	cout<<"BUILDING BSE..."<<endl;
	double* k_point_diff;
	complex<double> tmp_variable_1;
	complex<double> tmp_variable_2=-coulomb_potential->pull(excitonic_momentum);
	complex<double> tmp_variable_3;
	k_point_diff=new double[3];
	int position;

	//// parallelizing on the for loop calculations
	///#pragma omp parallel for
	for (int i=0;i<number_k_points_list;i++)
		for (int q=0;q<dimension_bse_hamiltonian_cv;q++)
			for (int j=0;j<number_k_points_list;j++){
				for(int r=0;r<3;r++)
					k_point_diff[r]=k_points_list[i][r]-k_points_list[j][r];
				tmp_variable_3=coulomb_potential->pull(k_point_diff);
				for (int s=0;s<dimension_bse_hamiltonian_cv;s++){
					//cout<<int(i/number_k_points_list*100)<<endl;
					tmp_variable_1=tmp_variable_2*function_exciton_product(excitonic_states[i][q],excitonic_states[j][s],number_wannier_functions,1)+tmp_variable_3*function_exciton_product(excitonic_states[i][q],excitonic_states[j][s],number_wannier_functions,0);
					position=s+dimension_bse_hamiltonian_cv*j+dimension_bse_hamiltonian*q+dimension_bse_hamiltonian*dimension_bse_hamiltonian_cv*i;
					excitonic_hamiltonian[position]=tmp_variable_1.real()+_Complex_I*(tmp_variable_1.imag());
				}
			}
			
	///adding the diagonal part to the BSE hamiltonian
	#pragma omp parallel for
	for (int i=0;i<number_k_points_list;i++)
		for (int q=0;q<dimension_bse_hamiltonian_cv;q++){
			position=q+dimension_bse_hamiltonian_cv*i+dimension_bse_hamiltonian*q+dimension_bse_hamiltonian*dimension_bse_hamiltonian_cv*i;
			excitonic_hamiltonian[position]=excitonic_hamiltonian[position]+excitonic_energies[i][q].real()+_Complex_I*(excitonic_energies[i][q].imag());
		}

	delete[] excitonic_energies;
	delete[] excitonic_states;
	delete[] ks_states_k_point;
	delete[] ks_states_k_point_q;
	delete[] ks_energies_k_point;
	delete[] ks_energies_k_point_q;
	delete[] ks_states_k_point_derivative;
	delete[] ks_energies_k_point_derivative;
	
	cout<<"BUILDING BSE FINISHED..."<<endl;

	return {dipoles, excitonic_hamiltonian};
};
/// //usual diagonalization routine
std:: tuple<complex<double>*,complex<double>**> Excitonic_Hamiltonian:: pull_eigenstates_usual_way(lapack_complex_double* excitonic_hamiltonian){

	/// diagonalizing the BSE matrix M_{(bz_number_k_points x number_valence_bands x number_conduction_bands)x(bz_number_k_points x number_valence_bands x number_conduction_bands)}
	int N=dimension_bse_hamiltonian; int LDA=dimension_bse_hamiltonian; int matrix_layout = 101; int INFO;
	lapack_complex_double *excitonic_eigenvalues_temporary;
	int LDVL=1;
	int LDVR=1;
	lapack_complex_double *VL;
	lapack_complex_double *VR; 
	char JOBVL = 'N'; char JOBVR = 'N';
	/// saving all the eigenvalues
	excitonic_eigenvalues_temporary=(lapack_complex_double*)malloc(dimension_bse_hamiltonian*sizeof(lapack_complex_double));

	INFO=LAPACKE_zgeev(matrix_layout,JOBVL,JOBVR,N,excitonic_hamiltonian,LDA,excitonic_eigenvalues_temporary,VL,LDVL,VR,LDVR);
	cout<<"INFO "<<INFO<<endl;
	complex<double> **excitonic_eigenstates;
	excitonic_eigenstates=new complex<double>*[dimension_bse_hamiltonian];
	for (int q=0; q<dimension_bse_hamiltonian;q++)
		excitonic_eigenstates[q]=new complex<double>[dimension_bse_hamiltonian];

	int counting;
	for (int q=0; q<dimension_bse_hamiltonian;q++){
		counting=0;
		for (int i=0;i<number_k_points_list;i++)
			for (int s=0; s<dimension_bse_hamiltonian_cv;s++){
				excitonic_eigenstates[q][counting]=excitonic_hamiltonian[q*dimension_bse_hamiltonian+counting];
				counting=counting+1;
			}
	}

	complex<double> *excitonic_eigenvalues;
	excitonic_eigenvalues=new complex<double>[dimension_bse_hamiltonian];
	complex<double> modulus;
	///normalizing eigenvectors and multiplying the eigenvalues to them
	for (int q=0; q<dimension_bse_hamiltonian;q++){
		modulus=sqrt(function_scalar_product(excitonic_eigenstates[q],excitonic_eigenstates[q],dimension_bse_hamiltonian));
		excitonic_eigenvalues[q]=excitonic_eigenvalues_temporary[q];
		if((modulus.real()!=0)||(modulus.imag()!=0))
			for(int j=0;j<dimension_bse_hamiltonian;j++)
				excitonic_eigenstates[q][j]=(excitonic_eigenstates[q][j]/modulus);
		else
			for(int j=0;j<dimension_bse_hamiltonian;j++)
				excitonic_eigenstates[q][j]=0.0;
	}
	
	free(excitonic_eigenvalues_temporary);

	return {excitonic_eigenvalues,excitonic_eigenstates};
};

/// FASTESTS diagonalization routine
/// Structure preserving parallel algorithms for solving the Bethe–Salpeter eigenvalue problem Meiyue Shao, Felipe H. da Jornada, Chao Yang, Jack Deslippe, Steven G. Louie
std::tuple<complex<double>*,complex<double>**> Excitonic_Hamiltonian:: pull_eigenstates_cholesky_way(lapack_complex_double* excitonic_hamiltonian){

	/// diagonalizing the BSE matrix M_{(bz_number_k_points x number_valence_bands x number_conduction_bands)x(bz_number_k_points x number_valence_bands x number_conduction_bands)}
	int dimension_bse_hamiltonian_2=dimension_bse_hamiltonian/2;
	complex<double> **A_matrix;
	complex<double> **B_matrix;
	A_matrix=new complex<double>*[dimension_bse_hamiltonian_2];
	B_matrix=new complex<double>*[dimension_bse_hamiltonian_2];
	for(int i=0;i<dimension_bse_hamiltonian_2;i++){
		A_matrix[i]=new complex<double>[dimension_bse_hamiltonian_2];
		B_matrix[i]=new complex<double>[dimension_bse_hamiltonian_2];
	}

	int counting1=0;
	int counting2;
	for(int k=0;k<number_k_points_list;k++)
		for(int q=0;q<dimension_bse_hamiltonian_cv;q++){
			counting2=0;
			for(int l=0;l<number_k_points_list;l++)
				for(int s=0;s<dimension_bse_hamiltonian_cv;s++){
					if((counting1<dimension_bse_hamiltonian_2)&&(counting2<dimension_bse_hamiltonian_2))
						A_matrix[counting1][counting2]=excitonic_hamiltonian[((k*number_k_points_list+q)*dimension_bse_hamiltonian_cv+l)*number_k_points_list+s];
					if((counting1<dimension_bse_hamiltonian_2)&&(counting2>=dimension_bse_hamiltonian_2))
						B_matrix[counting1][counting2-dimension_bse_hamiltonian_2]=excitonic_hamiltonian[((k*number_k_points_list+q)*dimension_bse_hamiltonian_cv+l)*number_k_points_list+s];
					counting2++;
				}
			counting1++;
		}

	///checking A and B matrices
	cout<<"A and B "<<endl;
	for (int q=0;q<dimension_bse_hamiltonian_2;q++){
		for (int s=0;s<dimension_bse_hamiltonian_2;s++)
			printf("(%6.2f+i%6.2f)|", A_matrix[q][s].real(),A_matrix[q][s].imag());
		for (int s=0;s<dimension_bse_hamiltonian_2;s++)
			printf("(%6.2f+i%6.2f)|", B_matrix[q][s].real(),B_matrix[q][s].imag());
		cout<<endl;
	}
	for (int q=0;q<dimension_bse_hamiltonian_2;q++){
		for (int s=0;s<dimension_bse_hamiltonian_2;s++)
			printf("(%6.2f+i%6.2f)|", -B_matrix[s][q].real(),B_matrix[s][q].imag());
		for (int s=0;s<dimension_bse_hamiltonian_2;s++)
			printf("(%6.2f+i%6.2f)|", -A_matrix[s][q].real(),-A_matrix[s][q].imag());
		cout<<endl;
	}

	complex<double>	AB_matrix_diff[dimension_bse_hamiltonian_2][dimension_bse_hamiltonian_2];
	complex<double>	AB_matrix_sum[dimension_bse_hamiltonian_2][dimension_bse_hamiltonian_2];
	for (int q=0; q<dimension_bse_hamiltonian_2;q++)
		for (int s=0; s<dimension_bse_hamiltonian_2;s++){
			AB_matrix_diff[q][s]=A_matrix[q][s]-B_matrix[q][s];
			AB_matrix_sum[q][s]=A_matrix[q][s]+B_matrix[q][s];
		}

	double* M_matrix;
	M_matrix=(double*)malloc(dimension_bse_hamiltonian*dimension_bse_hamiltonian*sizeof(double));
	
	for (int q=0;q<dimension_bse_hamiltonian;q++)
		for (int s=0;s<dimension_bse_hamiltonian;s++){
			if((q<dimension_bse_hamiltonian_2)&&(s<dimension_bse_hamiltonian_2))
				M_matrix[q*dimension_bse_hamiltonian+s]=AB_matrix_sum[q][s].real();
			else if ((q<dimension_bse_hamiltonian_2)&&(s>=dimension_bse_hamiltonian_2))
				M_matrix[q*dimension_bse_hamiltonian+s]=AB_matrix_diff[q][s-dimension_bse_hamiltonian_2].imag();
			else if ((q>=dimension_bse_hamiltonian_2)&&(s<dimension_bse_hamiltonian_2))
				M_matrix[q*dimension_bse_hamiltonian+s]=-AB_matrix_sum[q-dimension_bse_hamiltonian_2][s].imag();
			else
				M_matrix[q*dimension_bse_hamiltonian+s]=AB_matrix_diff[q-dimension_bse_hamiltonian_2][q-dimension_bse_hamiltonian_2].real();
		}
	
	cout<<"M matrix: "<<endl;
	for (int q=0;q<dimension_bse_hamiltonian;q++){
		for (int s=0;s<dimension_bse_hamiltonian;s++)
			printf("%6.2f ",M_matrix[q*dimension_bse_hamiltonian+s]);
		cout<<endl;
	}

	double* M_matrix_tmp;
	M_matrix_tmp=(double*)malloc(dimension_bse_hamiltonian*dimension_bse_hamiltonian*sizeof(double));
	for (int q=0;q<dimension_bse_hamiltonian;q++)
		for (int s=0;s<dimension_bse_hamiltonian;s++)
			M_matrix_tmp[q*dimension_bse_hamiltonian+s]=-M_matrix[q*dimension_bse_hamiltonian+s];

	///compute the Cholesky factorization
	char UPLO = 'L'; int N;	int LDA; int matrix_layout = 101; int INFO;
	LDA=dimension_bse_hamiltonian; N=dimension_bse_hamiltonian;
	INFO=LAPACKE_dpotrf(matrix_layout, UPLO, N, M_matrix, LDA);
	cout<<"Cholesky: "<<INFO<<endl;
	if (INFO>0){
		INFO=LAPACKE_dpotrf(matrix_layout, UPLO, N, M_matrix_tmp, LDA);
		cout<<"Cholesky second attempt: "<<INFO<<endl;
		for (int q=0;q<dimension_bse_hamiltonian;q++)
			for (int s=0;s<dimension_bse_hamiltonian;s++)
				M_matrix[q*dimension_bse_hamiltonian+s]=-M_matrix_tmp[q*dimension_bse_hamiltonian+s];
	}
	free(M_matrix_tmp);

	for(int q=0;q<dimension_bse_hamiltonian;q++)
		for(int s=0;s<dimension_bse_hamiltonian;s++)
			if(q<s)
				M_matrix[q*dimension_bse_hamiltonian+s]=0.0;
	///construct W
	double* W_matrix;
	W_matrix=(double*)malloc(dimension_bse_hamiltonian*dimension_bse_hamiltonian*sizeof(double));
	double J[dimension_bse_hamiltonian][dimension_bse_hamiltonian];
	for(int q=0;q<dimension_bse_hamiltonian;q++)
		for(int s=0;s<dimension_bse_hamiltonian;s++){
			W_matrix[q*dimension_bse_hamiltonian+s]=0.0;
			if((q<dimension_bse_hamiltonian_2)&&(s<dimension_bse_hamiltonian_2))
				J[q][s]=0.0;
			else if ((q<dimension_bse_hamiltonian_2)&&(s>=dimension_bse_hamiltonian_2))
				J[q][q]=1.0;
			else if ((q>=dimension_bse_hamiltonian_2)&&(s<dimension_bse_hamiltonian_2))
				J[q][q]=-1.0;
			else
				J[q][q]=0.0;
		}
	
	for(int q=0;q<dimension_bse_hamiltonian;q++)
		for(int s=0;s<dimension_bse_hamiltonian;s++)
			for(int t=0;t<dimension_bse_hamiltonian;t++)
				for(int v=0;v<dimension_bse_hamiltonian;v++)
					W_matrix[q*dimension_bse_hamiltonian+s]=W_matrix[q*dimension_bse_hamiltonian+s]+M_matrix[t*dimension_bse_hamiltonian+t]*J[t][v]*M_matrix[v*dimension_bse_hamiltonian+s];

	///diagonalizing the real values matrix
	char JOBVL='V'; char JOBVR='N'; int LDVR; int LDVL;
	double* VR;
	double eigenvalues_real[dimension_bse_hamiltonian];
	double eigenvalues_imag[dimension_bse_hamiltonian];
	double* eigenvectors;
	eigenvectors=(double*)malloc(dimension_bse_hamiltonian*dimension_bse_hamiltonian*sizeof(double));
	INFO=LAPACKE_dgeev(matrix_layout,JOBVL,JOBVR,N,W_matrix,LDA,eigenvalues_real,eigenvalues_imag,eigenvectors,LDVL,VR,LDVR);
	//INFO=LAPACKE_sgeevx()
	///obtaining the initial eigenvectors
	complex<double>* final_eigenvalues;
	complex<double>** final_eigenvectors;
	final_eigenvalues=new complex<double>[dimension_bse_hamiltonian];
	final_eigenvectors=new complex<double>*[dimension_bse_hamiltonian];
	for(int i=0;i<dimension_bse_hamiltonian;i++){
		final_eigenvectors[i]=new complex<double>[dimension_bse_hamiltonian];
		final_eigenvalues[i]=C(eigenvalues_real[i],0);
		for(int j=0;j<dimension_bse_hamiltonian;j++){
			final_eigenvectors[i][j].real(0.0);
			final_eigenvectors[i][j].imag(0.0);
		}
	}

	for(int i=0;i<dimension_bse_hamiltonian;i++)
		for(int j=0;j<dimension_bse_hamiltonian;j++)
			for(int k=0;k<dimension_bse_hamiltonian;k++)
				for(int r=0;r<dimension_bse_hamiltonian;r++)
					final_eigenvectors[i][j]=final_eigenvectors[i][j]+C(J[i][k]*M_matrix[k*dimension_bse_hamiltonian+r]*W_matrix[r*dimension_bse_hamiltonian+j],0)/final_eigenvalues[r];

	///normalizing the eigenvectors and equilizing the modulus to 1
	complex<double> modulus;
	for(int i=0;i<dimension_bse_hamiltonian;i++){
		modulus=sqrt(function_scalar_product(final_eigenvectors[i],final_eigenvectors[i],dimension_bse_hamiltonian));
		if((modulus.real()!=0)||(modulus.imag()!=0))
			for(int j=0;j<dimension_bse_hamiltonian;j++)
				final_eigenvectors[i][j]=(final_eigenvectors[i][j]/modulus);
		else
			for(int j=0;j<dimension_bse_hamiltonian;j++)
				final_eigenvectors[i][j]=0.0;
	}

	return {final_eigenvalues,final_eigenvectors};
};

complex<double>***Excitonic_Hamiltonian::pull_excitonic_oscillator_force(complex<double> **excitonic_eigenstates, complex<double>* excitonic_eigenergies, complex<double> **dipoles){
	complex<double>*** oscillator_force;
	oscillator_force=new complex<double>**[3];
	for(int i=0;i<3;i++){
		oscillator_force[i]=new complex<double>*[3];
		for(int j=0;j<3;j++)
			oscillator_force[i][j]=new complex<double>[dimension_bse_hamiltonian];
	}

	complex<double> temporary_sum1;
	complex<double> temporary_sum2;

	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			for(int q=0;q<dimension_bse_hamiltonian;q++){
				temporary_sum1.real(0.0); temporary_sum2.real(0.0);
				temporary_sum1.imag(0.0); temporary_sum2.imag(0.0);
				for(int r=0;r<dimension_bse_hamiltonian;r++){
					temporary_sum1=temporary_sum1+excitonic_eigenstates[q][r]*dipoles[i][r];
					temporary_sum2=temporary_sum2+conj(excitonic_eigenstates[q][r]*dipoles[j][r]);
				}
				//cout<<temporary_sum1<<"  "<<temporary_sum2<<endl;
				oscillator_force[i][j][q]=temporary_sum1*temporary_sum2;
			}
	
	return oscillator_force;
};

void Excitonic_Hamiltonian:: dielectric_tensor(double* excitonic_momentum, double eta, double *omega, int n, ofstream *file_diel, double scissor_operator){
	complex<double> **dipoles;
	lapack_complex_double *excitonic_hamiltonian;
	std::tuple<complex<double>**,lapack_complex_double*> dipoles_and_hamiltonian;
	dipoles_and_hamiltonian=pull_hamiltonian_and_renormalized_dipoles(excitonic_momentum);
	dipoles=std::get<0>(dipoles_and_hamiltonian);
	excitonic_hamiltonian=std::get<1>(dipoles_and_hamiltonian);
	complex<double> **eigenstates; complex<double> *eigenvalues;
	std::tuple<complex<double>*,complex<double>**> eigenvalues_and_eigenstates;
	eigenvalues_and_eigenstates=pull_eigenstates_usual_way(excitonic_hamiltonian);
	eigenvalues=std::get<0>(eigenvalues_and_eigenstates);
	eigenstates=std::get<1>(eigenvalues_and_eigenstates);
	complex<double> ***oscillator_force;
	oscillator_force=pull_excitonic_oscillator_force(eigenstates,eigenvalues,dipoles);
	
	double factor=pow(const_electron_charge,2)/(const_vacuum_dielectric_constant*coulomb_potential->pull_volume_unit_cell()*number_k_points_list);

	complex<double> **dielectric_function;
	dielectric_function=new complex<double>*[3];
	for(int i=0;i<3;i++){
		dielectric_function[i]=new complex<double>[3];
		for(int j=0;j<3;j++){
			dielectric_function[i][j].real(0.0);
			dielectric_function[i][j].imag(0.0);
		}
		dielectric_function[i][i].real(1.0);
	}
	
	double variable_tmp1;
	double variable_tmp2;
	for(int r=0;r<n;r++){
		omega[r]=omega[r]+scissor_operator/hbar;
		cout<<"status "<<double(r)/double(n)*100<<endl;
		for(int i=0;i<3;i++)
			for(int j=0;j<3;j++){
				for(int q=0;q<dimension_bse_hamiltonian;q++){
					//cout<<oscillator_force[i][j][q]<<endl;
					variable_tmp1=((oscillator_force[i][j][q]).real())/(pow((hbar*omega[r]-(eigenvalues[q].real())),2)+pow(eta,2));
					variable_tmp2=variable_tmp1*((eigenvalues[q].real())-hbar*omega[r]);
					//cout<<variable_tmp1<<variable_tmp2<<endl;
					dielectric_function[i][j].real(dielectric_function[i][j].real()+variable_tmp2);
					dielectric_function[i][j].imag(dielectric_function[i][j].imag()+variable_tmp1*eta);
					//cout<<oscillator_force[i][j][q]<<endl;
				}
				dielectric_function[i][j].real(factor*dielectric_function[i][j].real());
				dielectric_function[i][j].imag(factor*dielectric_function[i][j].imag());
				if(i==j){
					dielectric_function[i][j].real(1+dielectric_function[i][j].real());
					if(i==0)
						*file_diel<<omega[r]<<" "<<(dielectric_function[i][j]).imag()<<" ";
					else
					*file_diel<<(dielectric_function[i][j]).imag()<<" ";
				}
			}
		*file_diel<<endl;
	}

};

int main(){
	double fermi_energy=14.4239;
	K_points kpoint;
	//ifstream file_kpoint;
	//file_kpoint.open("k_points_list.dat");
	//int number_kpoint_list=5;
	//kpoint.push_list_values(&file_kpoint,number_kpoint_list);
	//file_kpoint.close();
	//double** kpoint_list=kpoint.pull_list_values();
	
	Crystal_Lattice crystal;
	ifstream file_crystal_bravais;
	file_crystal_bravais.open("bravais.lattice.data");
	ifstream file_crystal_coordinates;
	file_crystal_coordinates.open("atoms.data");
	crystal.push_values(&file_crystal_bravais,&file_crystal_coordinates);
	//crystal.print();

	double shift[3];
	shift[0]=0.0,shift[1]=0.0;shift[2]=0.0;
	kpoint.push_bz_values(&crystal,0.1,shift,1);
	double** kpoint_list=kpoint.pull_list_values();
	int number_kpoint_list=kpoint.pull_number_kpoint_list();
	kpoint.print_bz_values();
	cout<<"Number of K points "<<number_kpoint_list<<endl;

	ifstream file_htb;
	string seedname;
	file_htb.open("Calc_up_hr.dat");
	Hamiltonian_TB htb;
	htb.push_values(&file_htb,fermi_energy);
	//htb.print_hamiltonian();
	int number_of_valence_bands=2;
	int number_of_conduction_bands=2;
	htb.print_ks_states(kpoint_list[0],number_of_valence_bands,number_of_conduction_bands);

	double effective_dielectric_constant=1.0;
	double minimum_k_point_modulus=0.00001;
	int number_wannier_functions=htb.push_number_wannier_functions();
	double excitonic_momentum[3];
	excitonic_momentum[0]=0.0; excitonic_momentum[1]=0.0; excitonic_momentum[2]=0.0;
	Excitonic_Hamiltonian htbse;	
	htbse.push_values(number_of_valence_bands,number_of_conduction_bands,kpoint_list,number_kpoint_list,&file_htb,fermi_energy,crystal,effective_dielectric_constant,minimum_k_point_modulus,number_wannier_functions);
	//htbse.print(excitonic_momentum);
	
	double scissor_operator=0.00;
	ofstream file_diel;
	file_diel.open("BSE.dat");
	int n=400;
	double* omega;
	omega=new double[n];
	double eta=0.001;
	for(int i=0;i<n;i++)
		omega[i]=double(i/100);	
	htbse.dielectric_tensor(excitonic_momentum,eta,omega,n,&file_diel,scissor_operator);
	file_diel.close();

	return 0;
}
