#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <complex>
#include <cstdio>
#include <tuple>
#include <armadillo>
#include <omp.h>
#include <chrono>
#include <variant>
#include <mpi.h>
///#include <scalapackpp/wrappers/eigenvalue.hpp>
///#include <scalapackpp/scalapackpp.hpp>

using namespace std;
using namespace arma;
using two_or_one_mat = variant<tuple<cx_mat,cx_mat>, cx_mat>;

// declaring C function/libraries in the C++ code
extern "C"
{
// wrapper of the Fortran Lapack library into C
#include <stdio.h>
#include <omp.h>
#include <complex.h>
#include <lapacke.h>
}

///CONSTANT
const double minval = 1.0e-8;
const double pigreco = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;
///being bravais lattice in Ang, k points are in crystal coordinates and then transformed in Ang^{-1}
///it means that e^2/|k|^2 
/// 1/|k|^2 in natural units [(1.9733)^2x10^{-38} m^2] --> 1/|k|^2(Ang) x (1.9733)^2x10^{-18}
/// e^2 ->  (0.30282)^2
///const double conversionNmtoeV = pow(1.9733*0.30282,2)*10e-18;
const double conversionNmtoeV = pow(1.9733*0.30282,2)*10e-18;
const double hbar = 6.582119569;
const double hc = 911.38246268;
/// START DEFINITION DIFFERENT CLASSES

/// Crystal_Lattice class
class Crystal_Lattice
{
private:
	int number_atoms;
	double volume;
	mat atoms_coordinates;
	mat bravais_lattice{mat(3,3)};
	mat primitive_vectors{mat(3,3)};
public:
	Crystal_Lattice(string bravais_lattice_file_name, string atoms_coordinates_file_name,int number_atoms_tmp);
	vec pull_sitei_coordinates(int sitei);
	mat pull_bravais_lattice();
	mat pull_primitive_vectors();
	mat pull_atoms_coordinates();
	int pull_number_atoms();
	double pull_volume();
	void print();
	~Crystal_Lattice(){
		number_atoms=0;
		volume=0.0;
	};
};
int Crystal_Lattice::pull_number_atoms(){
	return number_atoms;
};
vec Crystal_Lattice::pull_sitei_coordinates(int sitei){
	return atoms_coordinates.col(sitei);
};
mat Crystal_Lattice::pull_bravais_lattice(){
	return bravais_lattice;
};
mat Crystal_Lattice::pull_atoms_coordinates(){
	return atoms_coordinates;
};
double Crystal_Lattice::pull_volume(){
	return volume;
};
mat Crystal_Lattice::pull_primitive_vectors(){
	return primitive_vectors;
};
Crystal_Lattice::Crystal_Lattice(string bravais_lattice_file_name,string atoms_coordinates_file_name,int number_atoms_tmp){
	ifstream bravais_lattice_file;
	bravais_lattice_file.open(bravais_lattice_file_name);
	ifstream atoms_coordinates_file;
	atoms_coordinates_file.open(atoms_coordinates_file_name);
	
	bravais_lattice_file.seekg(0);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			bravais_lattice_file >> bravais_lattice(i, j);
	
	number_atoms=number_atoms_tmp;
	atoms_coordinates.set_size(3,number_atoms);
	atoms_coordinates_file.seekg(0);
	for (int i=0;i<number_atoms;i++)
		for (int j=0;j<3;j++)
			atoms_coordinates_file>>atoms_coordinates(j, i);
			
	volume = arma::det(bravais_lattice);

	vec b0(3);
	vec b1(3);
	vec b2(3);
	for(int i=0;i<3;i++){
		b0(i)=bravais_lattice(i,0);
		b1(i)=bravais_lattice(i,1);
		b2(i)=bravais_lattice(i,2);
	}
	primitive_vectors.col(0) = cross(b1,b2);
	primitive_vectors.col(1) = cross(b2,b0);
	primitive_vectors.col(2) = cross(b0,b1);
	double factor = 2 * pigreco / volume;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			primitive_vectors(i, j) = factor*primitive_vectors(i, j);
	
	bravais_lattice_file.close();
	atoms_coordinates_file.close();
};
void Crystal_Lattice::print()
{
	cout << "Bravais Lattice:" << endl;
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			cout << bravais_lattice(i, j) << " ";
		cout << endl;
	}
	cout << "Atoms Coordinates:" << endl;
	for (int i = 0; i < number_atoms; i++)
	{
		for (int j = 0; j < 3; j++)
			cout << atoms_coordinates(j, i) << " ";
		cout << endl;
	}
};
/// K_points class
/// it is possible to define a list of k points directly from the BZ or as an input
/// in the class K_points the points of FBZ are saved as k_points_list, while the points outside of the FBZ, defining the rest of the reciprocal lattice, are saved as g_points_list
class K_points
{
private:
	double spacing;
	int number_k_points_list;
	int dimension;
	mat primitive_vectors{mat(3,3)};
	mat k_points_list;
	vec shift{vec(3)}; 
	vec direction_cutting{vec(3)};
public:
	K_points(Crystal_Lattice *crystal_lattice,vec shift_tmp);
	void push_k_points_list_values(string k_points_list_file_name,int number_k_points_list_tmp,int crystal_coordinates);
	void push_k_points_list_values(double spacing_tmp,int dimension_tmp,vec direction_cutting_tmp);
	int pull_number_k_points_list();
	mat pull_k_points_list_values();
	mat pull_primitive_vectors();
	vec pull_shift();
	void print();
	~K_points(){
		spacing=0;
		number_k_points_list=0;
	};
};
K_points::K_points(Crystal_Lattice *crystal_lattice,vec shift_tmp){
	shift=shift_tmp;
	primitive_vectors=crystal_lattice->pull_primitive_vectors();
};
mat K_points::pull_primitive_vectors(){
	return primitive_vectors;
};
void K_points::push_k_points_list_values(string k_points_list_file_name,int number_k_points_list_tmp,int crystal_coordinates){
	number_k_points_list = number_k_points_list_tmp;
	cout<<"number points "<<number_k_points_list<<endl;
	ifstream k_points_list_file;
	k_points_list_file.open(k_points_list_file_name);
	k_points_list_file.seekg(0);
	k_points_list.set_size(3, number_k_points_list);
	int counting = 0;
	while (k_points_list_file.peek()!=EOF){
		if (counting<number_k_points_list)
		{
			k_points_list_file >> k_points_list(0, counting);
			k_points_list_file >> k_points_list(1, counting);
			k_points_list_file >> k_points_list(2, counting);
			counting = counting + 1;
		}
		else
			///to avoid the reading of blank rows			
			break;
	}
	k_points_list_file.close();

	if(crystal_coordinates==1){
		mat k_points_list_tmp(3,number_k_points_list,fill::zeros);
		for(int i=0;i<number_k_points_list;i++)
			for(int s=0;s<3;s++){
				for(int r=0;r<3;r++)
					k_points_list_tmp(s,i)+=k_points_list(r,i)*primitive_vectors(s,r);
				k_points_list(s,i)=k_points_list_tmp(s,i);
			}
		k_points_list_tmp.reset();
	}
};
void K_points::push_k_points_list_values(double spacing_tmp,int dimension_tmp,vec direction_cutting_tmp){
	dimension=dimension_tmp;
	direction_cutting=direction_cutting_tmp;
	spacing=spacing_tmp;
	if(dimension==3){
		vec vec_number_k_points_list(3);
		for (int i = 0; i < 3; i++)
			vec_number_k_points_list(i) = int(sqrt(accu(primitive_vectors.col(i) % primitive_vectors.col(i))) / spacing);
		int limiti = int(vec_number_k_points_list(0));
		int limitj = int(vec_number_k_points_list(1));
		int limitk = int(vec_number_k_points_list(2));
		number_k_points_list = limiti * limitj * limitk;
		k_points_list.set_size(3,number_k_points_list);
		int counting = 0;
		for (int i = 0; i < limiti; i++)
			for (int j = 0; j < limitj; j++)
				for (int k = 0; k < limitk; k++){
					for (int r = 0; r < 3; r++)
						k_points_list(r, counting) = ((double)i / limiti) * (shift(r) + primitive_vectors(r, 0)) + ((double)j / limitj) * (shift(r) + primitive_vectors(r, 1)) + ((double)k / limitk) * (shift(r) + primitive_vectors(r, 2));
					counting = counting + 1;
				}
	}
	///TO IMPLEMENT OTHER CONDITIONS
};
vec K_points::pull_shift(){
	return shift;
};
mat K_points::pull_k_points_list_values(){
	return k_points_list;
};
int K_points::pull_number_k_points_list(){
	return number_k_points_list;
};
void K_points::print(){
	cout << "K points list" << endl;
	cout << number_k_points_list <<endl;
	for (int i = 0; i < number_k_points_list; i++)
	{
		cout << " ( ";
		for (int r = 0; r < 3; r++)
			cout << k_points_list(r, i) << " ";
		cout << " ) " << endl;
	}
	cout<<endl;
};
/// G_points class
/// the case one G point has to be properly implemented
class G_points
{
private:
	int number_g_points_list;
	vec number_g_points_direction;
	mat g_points_list;
	double cutoff_g_points_list;
	int dimension_g_points_list;
	vec direction_cutting{vec(3)};
	mat bravais_lattice{mat(3,3)};
	vec shift{vec(3)};
public:
	G_points(Crystal_Lattice *crystal_lattice,double cutoff_g_points_list_tmp,int dimension_g_points_list_tmp,vec direction_cutting_tmp,vec shift_tmp);
	mat pull_g_points_list_values();
	int pull_number_g_points_list();
	void print();
	~G_points(){
		number_g_points_list=0;
		cutoff_g_points_list=0.0;
		dimension_g_points_list=0;
	};
};
G_points::G_points(Crystal_Lattice *crystal_lattice,double cutoff_g_points_list_tmp,int dimension_g_points_list_tmp,vec direction_cutting_tmp,vec shift_tmp){
	dimension_g_points_list=dimension_g_points_list_tmp;
	direction_cutting=direction_cutting_tmp;
	cutoff_g_points_list=cutoff_g_points_list_tmp;
	shift=shift_tmp;

	mat primitive_vectors=crystal_lattice->pull_primitive_vectors();
	bravais_lattice=crystal_lattice->pull_bravais_lattice();
	cout<<"testing"<<endl;
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			cout<<dot(bravais_lattice.col(j),primitive_vectors.col(i))/(2*pigreco)<<endl;

	double max_g_value=cutoff_g_points_list;

	cout<<"Calculating g values..."<<endl;
	if(cutoff_g_points_list!=0){
		if(dimension_g_points_list==3){
			number_g_points_direction.zeros(3);
			number_g_points_list=1;
			for(int i=0;i<3;i++){
				number_g_points_direction(i)=int(max_g_value/norm(primitive_vectors.col(i),2));
				number_g_points_list=number_g_points_list*(2*number_g_points_direction(i)+1);
			}
			g_points_list.set_size(3,number_g_points_list);
			int counting=0;	
			for (int i = -number_g_points_direction(0); i <= number_g_points_direction(0); i++)
				for (int j = -number_g_points_direction(1); j <= number_g_points_direction(1); j++)
					for (int k = -number_g_points_direction(2); k <= number_g_points_direction(2); k++)
					{
						//if ((i!=0)||(j!=0)||(k!=0)){
							for (int r = 0; r < 3; r++)
								g_points_list(r, counting) =  i * (shift(r) + primitive_vectors(r, 0)) + j * (shift(r) + primitive_vectors(r, 1)) + k * (shift(r) + primitive_vectors(r, 2));
							counting = counting + 1;
						//}else
						//	for (int r = 0; r < 3; r++)
						//		g_points_list(r, 0) = 0.0;
							
					}
		}else if(dimension_g_points_list==2){
			number_g_points_direction.zeros(2);
			mat reciprocal_plane_along; reciprocal_plane_along.zeros(3,2);
			int counting=0;
			number_g_points_list=0;
			for(int i=0;i<3;i++)
				if(direction_cutting(i)==1){
					reciprocal_plane_along.col(counting)=primitive_vectors.col(i);
					number_g_points_direction(counting)=int(max_g_value/norm(primitive_vectors.col(i),2));
					number_g_points_list=number_g_points_list*(2*number_g_points_direction(i)+1);
					cout<<number_g_points_direction(counting)<<endl;
					counting++;
				}
			g_points_list.set_size(3,number_g_points_list);
			//cout << number_g_points_list << endl;
			counting = 1;
			for (int i = -number_g_points_direction(0); i <= number_g_points_direction(0); i++)
				for (int j = -number_g_points_direction(1); j <= number_g_points_direction(1); j++){
					//if((i!=0)||(j!=0)){
						for (int r = 0; r < 3; r++)
							g_points_list(r, counting) = i * (shift(r) + reciprocal_plane_along(r, 0)) + j * (shift(r) + reciprocal_plane_along(r, 1));
						//cout<<g_points_list.col(count)<<endl;
						counting = counting + 1;
					//}else
					//	for (int r = 0; r < 3; r++)
					//		g_points_list(r, 0) = 0.0;
				}
		}
		/// TO IMPLEMENT OTHER CASE
	}else{
		number_g_points_list=1;
		g_points_list.zeros(3,number_g_points_list);
	}
};
mat G_points::pull_g_points_list_values(){
	return g_points_list;
};
int G_points::pull_number_g_points_list(){
	return number_g_points_list;
};
void G_points::print(){
	cout << "G points list "<<number_g_points_list << endl;
	for (int i = 0; i < number_g_points_list; i++){
		cout << " ( ";
		for (int r = 0; r < 3; r++)
			cout << i<<" "<< g_points_list(r, i) << " ";
		cout << " ) " << endl;
	}
};
/// Hamiltonian_TB class
class Hamiltonian_TB
{
private:
	int spinorial_calculation;
	int number_wannier_functions;
	int htb_basis_dimension;
	int number_atoms;
	int number_primitive_cells;
	vec weights_primitive_cells;
	mat positions_primitive_cells;
	field<cx_cube> hamiltonian;
	field<mat> wannier_centers;
	bool dynamic_shifting;
	double fermi_energy;
	double little_shift;
	double scissor_operator;
	mat bravais_lattice;
public:
	Hamiltonian_TB(){
		number_wannier_functions = 0;
		htb_basis_dimension = 0;
		spinorial_calculation = 0;
		fermi_energy = 0;
		number_primitive_cells = 0;
		dynamic_shifting = false;
	};
	/// reading hamiltonian from wannier90 output
	Hamiltonian_TB(string wannier90_hr_file_name,string wannier90_centers_file_name,double fermi_energy_tmp,int spinorial_calculation_tmp,int number_atoms_tmp,bool dynamic_shifting_tmp,double little_shift_tmp,double scissor_operator_tmp,mat bravais_lattice_tmp);
	field<cx_mat> FFT(vec k_point);
	tuple<mat,cx_mat> pull_ks_states(vec k_point);
	tuple<mat,cx_mat> pull_ks_states_subset(vec k_point,int number_valence_bands_selected,int number_conduction_bands_selected);
	field<cx_cube> pull_hamiltonian();
	int pull_htb_basis_dimension();
	int pull_number_wannier_functions();
	double pull_fermi_energy();
	void print_hamiltonian();
	void print_ks_states(vec k_point, int number_valence_bands_selected, int number_conduction_bands_selected);
	field<mat> pull_wannier_centers();
	void print();
	~Hamiltonian_TB(){
		number_wannier_functions = 0;
		htb_basis_dimension = 0;
		spinorial_calculation = 0;
		fermi_energy = 0;
		number_primitive_cells = 0;
		dynamic_shifting = false;
	};
};
Hamiltonian_TB::Hamiltonian_TB(string wannier90_hr_file_name,string wannier90_centers_file_name,double fermi_energy_tmp,int spinorial_calculation_tmp,int number_atoms_tmp,bool dynamic_shifting_tmp,double little_shift_tmp,double scissor_operator_tmp,mat bravais_lattice_tmp){
	cout<<"Be Carefull: if you are doing a collinear spin calculation, the number of Wannier functions in the two spin channels has to be the same!!"<<endl;
	fermi_energy=fermi_energy_tmp;
	number_atoms=number_atoms_tmp;
	spinorial_calculation=spinorial_calculation_tmp;
	dynamic_shifting=dynamic_shifting_tmp;
	scissor_operator=scissor_operator_tmp;
	bravais_lattice=bravais_lattice_tmp;
	ifstream wannier90_hr_file;
	ifstream wannier90_centers_file;
	wannier90_hr_file.open(wannier90_hr_file_name);
	wannier90_centers_file.open(wannier90_centers_file_name);

	cout<<"Reading Hamiltonian..."<<endl;
	int total_elements;
	string history_time;
	int counting_primitive_cells;
	int counting_positions;
	int l;	int m;
	double trashing_positions[3];
	string trashing_lines;
	double real_part;
	double imag_part;
	int spin_channel = 0;
	wannier90_hr_file.seekg(0);
	/// the Hamiltonians for the spinorial calculation = 1, should be one under the other(all the hr FILE (time included))
	while (wannier90_hr_file.peek() != EOF && spin_channel < 2){
		getline(wannier90_hr_file >> ws, history_time);
		wannier90_hr_file >> number_wannier_functions;
		wannier90_hr_file >> number_primitive_cells;
		cout<<"Number wannier functions "<<number_wannier_functions<<endl;
		cout<<"Number primitive cells "<<number_primitive_cells<<endl;
		if (spin_channel == 0){
			// initialization of the vriables
			weights_primitive_cells.set_size(number_primitive_cells);
			positions_primitive_cells.set_size(3, number_primitive_cells);
			if (spinorial_calculation == 1){
				/// two loops
				hamiltonian.set_size(2);
				hamiltonian(0).set_size(number_wannier_functions, number_wannier_functions, number_primitive_cells);
				hamiltonian(1).set_size(number_wannier_functions, number_wannier_functions, number_primitive_cells);
				htb_basis_dimension = number_wannier_functions * 2;
			}
			else{
				/// one loop
				hamiltonian.set_size(1);
				hamiltonian(0).set_size(number_wannier_functions, number_wannier_functions, number_primitive_cells);
				htb_basis_dimension = number_wannier_functions;
			}
		}
		total_elements = number_wannier_functions * number_wannier_functions * number_primitive_cells;
		//cout<<"Total elements "<<total_elements<<endl;
		counting_positions = 0;
		while (counting_positions < number_primitive_cells){
			wannier90_hr_file >> weights_primitive_cells(counting_positions);
			//cout<<counting_positions<<" "<<weights_primitive_cells(counting_positions)<<endl;
			counting_positions++;
		}
		counting_primitive_cells = 0;
		counting_positions = 0;
		/// the hamiltonian in the collinear case is diagonal in the spin channel
		while (counting_positions < total_elements){
			if (counting_positions == number_wannier_functions * number_wannier_functions * counting_primitive_cells)
			{
				wannier90_hr_file >> positions_primitive_cells(0, counting_primitive_cells) >> positions_primitive_cells(1, counting_primitive_cells) >> positions_primitive_cells(2, counting_primitive_cells);	
				counting_primitive_cells = counting_primitive_cells + 1;
			}
			else
				wannier90_hr_file >> trashing_positions[0] >> trashing_positions[1] >> trashing_positions[2];
			wannier90_hr_file >> l >> m;
			wannier90_hr_file >> real_part >> imag_part;

			hamiltonian(spin_channel)(l - 1, m - 1, counting_primitive_cells - 1).real(real_part);
			hamiltonian(spin_channel)(l - 1, m - 1, counting_primitive_cells - 1).imag(imag_part);
			counting_positions++;
		}
		///cout<<spin_channel<<" "<<l<<" "<<m<<" "<<real(hamiltonian(spin_channel).tube(l-1, m-1))<<" "<<imag(hamiltonian(spin_channel).tube(l-1, m-1))<<" "<<counting_positions<<" "<<total_elements<<" "<<counting_primitive_cells<<" "<<number_primitive_cells<<endl;
		///cout<<spinorial_calculation<<" "<<number_wannier_functions<<" "<<number_primitive_cells<<" "<<spin_channel<<" "<<endl;
		if (spinorial_calculation==1)
			spin_channel = spin_channel + 1;
		else
			spin_channel=2;
	}
	cout<<"Hamiltonian saved."<<endl;
	cout<<"Converting positions primitive cells from crystal to cartesian coordinates"<<endl;
	mat positions_primitive_cells_tmp(3,number_primitive_cells,fill::zeros);
	for(int i=0;i<number_primitive_cells;i++)
		for(int s=0;s<3;s++){
			for(int r=0;r<3;r++)
				positions_primitive_cells_tmp(s,i)+=positions_primitive_cells(r,i)*bravais_lattice(s,r);
			positions_primitive_cells(s,i)=positions_primitive_cells_tmp(s,i);
		}
	positions_primitive_cells_tmp.reset();
	//if (wannier90_centers_file == NULL)
	//	throw std::invalid_argument("No Wannier90 Centers file!");
	//else
	cout<<"Reading centers..."<<endl;
	char element_name;
	int number_lines;
	spin_channel=0;
	wannier90_centers_file.seekg(0);
	while (wannier90_centers_file.peek() != EOF && spin_channel<2){
		if (spin_channel == 0){
			// initialization of the variables
			if (spinorial_calculation == 1){
				/// two loops
				wannier_centers.set_size(2);
				wannier_centers(0).set_size(3,number_wannier_functions);
				wannier_centers(1).set_size(3,number_wannier_functions);
			}
			else{
				/// one loop
				wannier_centers.set_size(1);
				wannier_centers(0).set_size(3,number_wannier_functions);
			}
		}

		wannier90_centers_file >> number_lines;
		getline(wannier90_centers_file >> ws, history_time);
		counting_positions = 0;
		while (counting_positions < number_wannier_functions){
			wannier90_centers_file >> element_name >> wannier_centers(spin_channel)(0,counting_positions) >> wannier_centers(spin_channel)(1,counting_positions) >> wannier_centers(spin_channel)(2,counting_positions);
			//cout<<spin_channel<<" "<<counting_positions<<" "<<wannier_centers(spin_channel)(0,counting_positions)<<" "<<wannier_centers(spin_channel)(1,counting_positions)<<" "<<wannier_centers(spin_channel)(2,counting_positions)<<endl;
			counting_positions++;
		}
		counting_positions = 0;
		while (counting_positions < number_atoms){
			getline(wannier90_centers_file >> ws, trashing_lines);
			counting_positions++;
		}
		//cout<<spin_channel<<" "<<number_atoms<<endl;
		spin_channel++;
	}
	cout<<"Centers saved."<<endl;
	wannier90_hr_file.close();
	wannier90_centers_file.close();


};
field<cx_cube> Hamiltonian_TB::pull_hamiltonian(){
	return hamiltonian;
};
int Hamiltonian_TB::pull_htb_basis_dimension(){
	return htb_basis_dimension;
};
int Hamiltonian_TB::pull_number_wannier_functions(){
	return number_wannier_functions;
};
double Hamiltonian_TB::pull_fermi_energy(){
	return fermi_energy;
};
void Hamiltonian_TB::print(){
	cout<<"Printing Hamiltonian..."<<endl;
	int spin_counting = 0;
	while (spin_counting < 2){
		for (int i = 0; i < number_primitive_cells; i++)
			for (int q = 0; q < number_wannier_functions; q++)
				for (int s = 0; s < number_wannier_functions; s++){
					for (int r = 0; r < 3; r++)
						cout << positions_primitive_cells(r, i) << " ";
					cout << s << " " << q << " " << hamiltonian(spin_counting)(s, q, i) << endl;
				}
		cout<<"Printing Wannier Centers..."<<endl;
		for(int i=0;i<number_wannier_functions;i++){
			for (int r = 0; r < 3; r++)
				cout << wannier_centers(spin_counting)(r,i) <<" ";
			cout<<endl;
		}
		if (spinorial_calculation == 1)
			spin_counting = spin_counting + 1;
		else
			spin_counting = 2;
	}
};
field<mat> Hamiltonian_TB::pull_wannier_centers(){
	return wannier_centers;
};
field<cx_mat> Hamiltonian_TB::FFT(vec k_point){
	int flag_spin_channel = 0;
	int offset;
	
	vec temporary_cos(number_primitive_cells);
	vec temporary_sin(number_primitive_cells);
	vec real_part_hamiltonian(number_primitive_cells);
	vec imag_part_hamiltonian(number_primitive_cells);
	//#pragma omp parallel for 
	for (int r = 0; r < number_primitive_cells; r++){
		temporary_cos(r) = cos(accu(k_point % positions_primitive_cells.col(r)));
		temporary_sin(r) = sin(accu(k_point % positions_primitive_cells.col(r)));
		///cout<<r<<positions_primitive_cells.col(r)<<" "<<k_point<<" "<<temporary_cos(r)<<" "<<temporary_sin(r)<<endl;
	}
	if (spinorial_calculation == 1){
		field<cx_mat> fft_hamiltonian(2);
		fft_hamiltonian(0).zeros(number_wannier_functions,number_wannier_functions);
		fft_hamiltonian(1).zeros(number_wannier_functions,number_wannier_functions);
		while (flag_spin_channel < 2){
			offset = number_wannier_functions * flag_spin_channel;
			//#pragma omp parallel for collapse(2) private(real_part_hamiltonian,imag_part_hamiltonian)
			for (int l = 0; l < number_wannier_functions; l++)
				for (int m = 0; m < number_wannier_functions; m++){
					real_part_hamiltonian = real(hamiltonian(flag_spin_channel).tube(l, m));
					imag_part_hamiltonian = imag(hamiltonian(flag_spin_channel).tube(l, m));
					real_part_hamiltonian = real_part_hamiltonian%weights_primitive_cells;
					imag_part_hamiltonian = imag_part_hamiltonian%weights_primitive_cells;
					fft_hamiltonian(flag_spin_channel)(l, m).real(accu(real_part_hamiltonian % temporary_cos) - accu(imag_part_hamiltonian % temporary_sin));
					fft_hamiltonian(flag_spin_channel)(l, m).imag(accu(imag_part_hamiltonian % temporary_cos) + accu(real_part_hamiltonian % temporary_sin));
				}
			//cout<<flag_spin_channel<<" "<<fft_hamiltonian(flag_spin_channel)<<endl;
			flag_spin_channel++;
		}
		return fft_hamiltonian;
	}else{
		field<cx_mat> fft_hamiltonian(1);
		fft_hamiltonian(0).zeros(number_wannier_functions, number_wannier_functions);
		//#pragma omp parallel for collapse(2) private(real_part_hamiltonian,imag_part_hamiltonian)
		for (int l = 0; l < number_wannier_functions; l++)
		{
			for (int m = 0; m < number_wannier_functions; m++)
			{
				///cout<<l<<m<<endl;
				real_part_hamiltonian = real(hamiltonian(0).tube(l, m));
				imag_part_hamiltonian = imag(hamiltonian(0).tube(l, m));
				real_part_hamiltonian = real_part_hamiltonian%weights_primitive_cells;
				imag_part_hamiltonian = imag_part_hamiltonian%weights_primitive_cells;
				fft_hamiltonian(0)(l, m).real(accu(real_part_hamiltonian % temporary_cos) - accu(imag_part_hamiltonian % temporary_sin));
				fft_hamiltonian(0)(l, m).imag(accu(imag_part_hamiltonian % temporary_cos) + accu(real_part_hamiltonian % temporary_sin));
			}
		}
		//cout<<"end fft"<<endl;
		return fft_hamiltonian;
	}	
};
tuple<mat, cx_mat> Hamiltonian_TB::pull_ks_states(vec k_point){
	/// the eigenvalues are saved into a two component element, in order to make the code more general
	mat ks_eigenvalues_spinor(2, number_wannier_functions, fill::zeros);
	cx_mat ks_eigenvectors_spinor(htb_basis_dimension, number_wannier_functions, fill::zeros);

	////field<cx_mat> fft_hamiltonian = FFT(k_point);
	if (spinorial_calculation == 1){
		cx_mat hamiltonian_up(number_wannier_functions, number_wannier_functions);
		cx_mat hamiltonian_down(number_wannier_functions, number_wannier_functions);
		field<cx_mat> fft_hamiltonian = FFT(k_point);
		hamiltonian_up = fft_hamiltonian(0);
		hamiltonian_down = fft_hamiltonian(1);
		//cout<<"diagonanlization starting"<<endl;
		vec eigenvalues_up(number_wannier_functions);
		cx_mat eigenvectors_up(number_wannier_functions,number_wannier_functions);
		vec eigenvalues_down(number_wannier_functions);
		cx_mat eigenvectors_down(number_wannier_functions,number_wannier_functions);
		//ARMADILLO DIAGONALIZATION ROUTINE BADLY FAILING; USING THE LAPACKE DIAGONALIZATION ROUTINE, INSTEAD
		//eig_gen(eigenvalues_up,eigenvectors_up, hamiltonian_up);
		//eig_gen(eigenvalues_down,eigenvectors_down, hamiltonian_down);
		lapack_complex_double *temporary_up; temporary_up = (lapack_complex_double *)malloc(number_wannier_functions*number_wannier_functions*sizeof(lapack_complex_double));
		lapack_complex_double *temporary_down; temporary_down = (lapack_complex_double *)malloc(number_wannier_functions*number_wannier_functions*sizeof(lapack_complex_double));

		#pragma omp parallel for collapse(2)
		for(int i=0;i<number_wannier_functions;i++)
			for(int j=0;j<number_wannier_functions;j++){
				temporary_up[i*number_wannier_functions+j]=real(hamiltonian_up(i,j))+_Complex_I*imag(hamiltonian_up(i,j));
				temporary_down[i*number_wannier_functions+j]=real(hamiltonian_down(i,j))+_Complex_I*imag(hamiltonian_down(i,j));
			}
	
		int N=number_wannier_functions;
		int LDA=number_wannier_functions;
		int matrix_layout = 101;
		int INFO;
		double *w_up; complex<double> **u_up;
		double *w_down; complex<double> **u_down;
		char JOBZ = 'V'; char UPLO = 'L';
		//// saving all the eigenvalues
		w_up = (double *)malloc(number_wannier_functions * sizeof(double));
		w_down = (double *)malloc(number_wannier_functions * sizeof(double));
		INFO = LAPACKE_zheev(matrix_layout, JOBZ, UPLO, N, temporary_up, LDA, w_up);
		INFO = LAPACKE_zheev(matrix_layout, JOBZ, UPLO, N, temporary_down, LDA, w_down);
		#pragma omp parallel for collapse(2)
		for(int i=0;i<number_wannier_functions;i++){
			for(int j=0;j<number_wannier_functions;j++){
				eigenvectors_up(i,j)=lapack_complex_double_real(temporary_up[i*number_wannier_functions+j])+_Complex_I*lapack_complex_double_imag(temporary_up[i*number_wannier_functions+j]);
				eigenvectors_down(i,j)=lapack_complex_double_real(temporary_down[i*number_wannier_functions+j])+_Complex_I*lapack_complex_double_imag(temporary_down[i*number_wannier_functions+j]);
			}
		}
		for(int i=0;i<number_wannier_functions;i++){
			eigenvalues_up(i)=w_up[i];
			eigenvalues_down(i)=w_down[i];
		}
		uvec ordering_up = sort_index(real(eigenvalues_up));
		uvec ordering_down = sort_index(real(eigenvalues_down));
		/// in the case of spinorial_calculation=1 combining the two components of spin into a single spinor
		/// saving the ordered eigenvectors in the matrix ks_eigenvectors_spinor
		for (int i = 0; i < number_wannier_functions; i++){
			for (int j = 0; j < htb_basis_dimension; j++){
				if (j < number_wannier_functions)
					ks_eigenvectors_spinor(j, i) = eigenvectors_up(j, ordering_up(i));
				else
					ks_eigenvectors_spinor(j, i) = eigenvectors_down(j - number_wannier_functions, ordering_down(i));
			}
			ks_eigenvectors_spinor.col(i) = ks_eigenvectors_spinor.col(i) / norm(ks_eigenvectors_spinor.col(i), 2);
			for (int r = 0; r < 2; r++)
				ks_eigenvalues_spinor(r, i) = (1 - r) * real(eigenvalues_up(ordering_up(i))) + r * real(eigenvalues_down(ordering_down(i)));
		}
		//cout<<"diagonanlization ending"<<endl;
		return {ks_eigenvalues_spinor, ks_eigenvectors_spinor};
	
	}else{
		field<cx_mat> fft_hamiltonian = FFT(k_point);
		
		int N=number_wannier_functions;
		int LDA=number_wannier_functions;
		int matrix_layout = 101;
		int INFO;
		double *w; complex<double> **u;
		char JOBZ = 'V'; char UPLO = 'L';

		//// saving all the eigenvalues
		w = (double *)malloc(number_wannier_functions * sizeof(double));
		
		lapack_complex_double *temporary; temporary = (lapack_complex_double *)malloc(number_wannier_functions*number_wannier_functions*sizeof(lapack_complex_double));
		#pragma omp parallel for collapse(2)
		for(int i=0;i<number_wannier_functions;i++)
			for(int j=0;j<number_wannier_functions;j++){
				temporary[i*number_wannier_functions+j]=real(fft_hamiltonian(0)(i,j))+_Complex_I*imag(fft_hamiltonian(0)(i,j));
			}
		
		INFO = LAPACKE_zheev(matrix_layout, JOBZ, UPLO, N, temporary, LDA, w);

		cx_vec eigenvalues(number_wannier_functions);
		cx_mat eigenvectors(number_wannier_functions,number_wannier_functions);

		#pragma omp parallel for collapse(2)
		for(int i=0;i<number_wannier_functions;i++)
			for(int j=0;j<number_wannier_functions;j++)
				eigenvectors(i,j)=lapack_complex_double_real(temporary[i*number_wannier_functions+j])+_Complex_I*lapack_complex_double_imag(temporary[i*number_wannier_functions+j]);

		for(int i=0;i<number_wannier_functions;i++)
			eigenvalues(i)=w[i];

		free(w); free(temporary);

		uvec ordering = sort_index(real(eigenvalues));

		/// in the case of spinorial_calculation=1 combining the two components of spin into a single spinor
		/// saving the ordered eigenvectors in the matrix ks_eigenvectors_spinor
		#pragma omp parallel for collapse(2)
		for (int i = 0; i < number_wannier_functions; i++)
			for (int j = 0; j < htb_basis_dimension; j++){
				ks_eigenvectors_spinor(j, i) = eigenvectors(j, ordering(i))/norm(eigenvectors.col(ordering(i)), 2);
			}

		for (int i = 0; i < number_wannier_functions; i++){
			ks_eigenvalues_spinor(0,i)=real(eigenvalues(ordering(i)));
			ks_eigenvalues_spinor(1,i)=real(eigenvalues(ordering(i)));
		}

		return {ks_eigenvalues_spinor, ks_eigenvectors_spinor};
	}
};
tuple<mat, cx_mat> Hamiltonian_TB::pull_ks_states_subset(vec k_point,int number_valence_bands_selected,int number_conduction_bands_selected){
	int number_valence_bands = 0;
	int number_conduction_bands = 0;
	int dimensions_subspace = number_conduction_bands_selected + number_valence_bands_selected;
	vec spinor_scissor_operator(2);
	spinor_scissor_operator(0)=scissor_operator;
	spinor_scissor_operator(1)=scissor_operator;
	
	mat ks_eigenvalues(2,number_wannier_functions, fill::zeros);
	cx_mat ks_eigenvectors(htb_basis_dimension, number_wannier_functions, fill::zeros);
	tuple<mat,cx_mat> ks_states(ks_eigenvalues,ks_eigenvectors);
	ks_states=pull_ks_states(k_point);
	ks_eigenvalues=get<0>(ks_states);
	ks_eigenvectors=get<1>(ks_states);

	/// distinguishing between valence and conduction states
	for (int i = 0; i < number_wannier_functions; i++){
		///cout<<ks_eigenvalues(0, i)<<" "<<ks_eigenvalues(1, i)<<endl; 
		if (ks_eigenvalues(0, i)<=fermi_energy && ks_eigenvalues(1, i)<=fermi_energy)
			number_valence_bands++;
		else
			number_conduction_bands++;
	}
	///cout<<"Number valence bands "<<number_valence_bands<<" Number conduction bands "<<number_conduction_bands<<endl;
	
	/// in a single matrix: first are written valence states, than (at higher rows) conduction states
	mat ks_eigenvalues_subset(2, dimensions_subspace);
	cx_mat ks_eigenvectors_subset(htb_basis_dimension, dimensions_subspace);
	for (int i = 0; i < dimensions_subspace; i++){
		//cout<<ks_eigenvalues.col(i)<<endl;
		if (i < number_valence_bands_selected){
			ks_eigenvectors_subset.col(i) = ks_eigenvectors.col((number_valence_bands - 1) - i);
			ks_eigenvalues_subset.col(i) = ks_eigenvalues.col((number_valence_bands - 1) - i);
		}else{
			ks_eigenvectors_subset.col(i) = ks_eigenvectors.col(number_valence_bands + (i - number_valence_bands_selected));
			ks_eigenvalues_subset.col(i) = ks_eigenvalues.col(number_valence_bands + (i - number_valence_bands_selected))+ spinor_scissor_operator;
		}
	}
	
	return {ks_eigenvalues_subset, ks_eigenvectors_subset};
};
void Hamiltonian_TB::print_ks_states(vec k_point, int number_valence_bands_selected, int number_conduction_bands_selected){
	tuple<mat, cx_mat> results_htb;
	tuple<mat, cx_mat> results_htb_subset;
	cout<<"Extraction all ks states:"<<endl;
	results_htb=pull_ks_states(k_point);
	cout<<"Extraction subset ks states"<<endl;
	results_htb_subset=pull_ks_states_subset(k_point,number_valence_bands_selected,number_conduction_bands_selected);
	mat eigenvalues=get<0>(results_htb);
	cx_mat eigenvectors=get<1>(results_htb);
	mat eigenvalues_subset=get<0>(results_htb_subset);
	cx_mat eigenvectors_subset=get<1>(results_htb_subset);
	////In the case of spinorial_calculation=1, the spinor components of each wannier function are one afte the other
	////i.e. wannier function 0 : spin up component columnt 0, spin down component column 1...and so on...
	/////moreover, in the case of ks_subset the valence states are written before the conduction states
	cout<<"All bands"<<endl;
	for (int i = 0; i < number_wannier_functions; i++){
		printf("%d	%.5f %.5f\n", i, eigenvalues(0, i), eigenvalues(1, i));
		//for (int j = 0; j < htb_basis_dimension; j++)
		//	printf("(%.5f,%.5f)", real(eigenvectors(j, i)), imag(eigenvectors(j, i)));
		cout << endl;
	}
	cout << "Only subset" << endl;
	for (int i = 0; i < number_valence_bands_selected + number_conduction_bands_selected; i++){
		printf("%d	%.5f %.5f\n", i, eigenvalues_subset(0, i), eigenvalues_subset(1, i));
		//for (int j = 0; j < htb_basis_dimension; j++)
			//printf("(%.5f,%.5f)", real(eigenvectors_subset(j, i)), imag(eigenvectors_subset(j, i)));
		cout << endl;
	}
};

// Coulomb_Potential class
class Coulomb_Potential
{
private:
	double minimum_k_point_modulus;
	mat primitive_vectors{mat(3,3)};
	vec direction_cutting{vec(3)};
	int dimension_potential;
	K_points *k_points;
	G_points *g_points;
	double volume_cell;
	double radius;
public:
	Coulomb_Potential(K_points* k_points_tmp,G_points* g_points_tmp,double minimum_k_point_modulus_tmp,int dimension_potential_tmp,vec direction_cutting_tmp,double volume_tmp,double radius_tmp);
	double pull(vec k_point);
	double pull_volume();
	void print();
	void print_profile(int number_k_points,double max_radius,string file_coulomb_potential_name, int direction_profile_xyz);
	///~Coulomb_Potential(){
	///	k_points=NULL;
	///	g_points=NULL;
	///};
};
Coulomb_Potential::Coulomb_Potential(K_points* k_points_tmp,G_points* g_points_tmp,double minimum_k_point_modulus_tmp,int dimension_potential_tmp,vec direction_cutting_tmp,double volume_tmp,double radius_tmp){
	k_points = k_points_tmp;
	g_points = g_points_tmp;
	minimum_k_point_modulus = minimum_k_point_modulus_tmp;
	radius=radius_tmp;
	dimension_potential = dimension_potential_tmp;
	///direction of the bravais lattice along whic the cut is considered (1 is cut and 0 is no-cut)
	direction_cutting = direction_cutting_tmp;
	primitive_vectors = k_points->pull_primitive_vectors();
	volume_cell=volume_tmp;
};
double Coulomb_Potential::pull_volume(){
	return volume_cell;
};
void Coulomb_Potential::print(){
	cout<<"Minimum k point modulus: "<<minimum_k_point_modulus<<endl;
	cout<<"Primitive vectors: "<<endl;
	cout<<primitive_vectors<<endl;
	cout<<"Direction cutting "<<endl;
	cout<<direction_cutting<<endl;
	cout<<"Dimension potential: "<<dimension_potential<<endl;
	cout<<"K points address: "<<k_points<<endl;
};
void Coulomb_Potential:: print_profile(int number_k_points,double max_k_point,string file_coulomb_potential_name, int direction_profile_xyz){
	vec k_point(3,fill::zeros);
	ofstream file_coulomb_potential;
	file_coulomb_potential.open(file_coulomb_potential_name);
	double coulomb;
	for(int i=0;i<number_k_points;i++){
		k_point(direction_profile_xyz)=(double(i)/double(number_k_points))*max_k_point;
		coulomb=pull(k_point);
		file_coulomb_potential<<i<<" "<<k_point(direction_profile_xyz)<<" "<<coulomb<<endl;
		k_point(direction_profile_xyz)=0.0;
	}
	file_coulomb_potential.close();
};
double Coulomb_Potential::pull(vec k_point){
	/// the volume of the cell is in angstrom
	/// the momentum k is in angstrom^-1
	double coulomb_potential;

	/// a cutoff is introduced in order to avoid any divergence in k=0
	double modulus_k_point;
	modulus_k_point= accu(k_point%k_point);
	if(dimension_potential==3){
		if (modulus_k_point < minimum_k_point_modulus)
			coulomb_potential = 0;
		else
			coulomb_potential = -conversionNmtoeV*4*pigreco/(modulus_k_point+radius);
	}else if(dimension_potential==2){
		vec primitive_along(3);
		for(int i=0;i<3;i++)
			if(direction_cutting(i)==0){
				primitive_along=primitive_vectors.col(i);
			}
		vec k_point_orthogonal(3,fill::zeros);
		vec k_point_along(3,fill::zeros);
		vec unity(3,fill::ones);
		k_point_along=(primitive_along%k_point)/norm(primitive_along);
		k_point_orthogonal=k_point-k_point_along;
		double c1=norm(k_point_along)/norm(k_point_orthogonal);
		double c2=(norm(primitive_along)/2)*norm(k_point_orthogonal);
		double c3=(norm(primitive_along)/2)*norm(k_point_along);
		if (modulus_k_point < minimum_k_point_modulus)
			coulomb_potential=0;
		else{
			coulomb_potential = -conversionNmtoeV*4*pigreco/(pow(modulus_k_point, 2)+radius);
			coulomb_potential = coulomb_potential*(1-exp(-c2)*(c1*sin(c3)-cos(c3)));
		}
	}
	///TO IMPLEMENT THIS CASE AS WELL
	return coulomb_potential;
};

/// Generalized dipoles elements
///rho_{n1,n2,k1-p,k2-q}(excitonic_momentum,G)=\bra{n1k1-p}e^{i(excitonic_momentum+G)r\ket{n2k2-q}
/// the shape of the dipoles has been chosen in order to facilitate calculations with supercell with significant local-field effects
class Dipole_Elements
{
private:
	int number_k_points_list;
	int number_g_points_list;
	mat k_points_list;
	mat g_points_list;
	int htb_basis_dimension;
	int spin_htb_basis_dimension;
	int number_wannier_centers;
	field<mat> wannier_centers;
	int number_valence_bands;
	int spin_number_valence_bands;
	int number_conduction_bands;
	int spin_number_conduction_bands;
	int number_valence_plus_conduction;
	int spin_number_valence_plus_conduction;
	Hamiltonian_TB *hamiltonian_tb;
	int spinorial_calculation;
	umat pair_k_points;
public:
	Dipole_Elements(int number_k_points_list_tmp,mat k_points_list_tmp, int number_g_points_list_tmp, mat g_points_list_tmp,int number_wannier_centers_tmp,int number_valence_bands_selected_tmp,int number_conduction_bands_selected_tmp, Hamiltonian_TB *hamiltonian_tb_tmp,int spinorial_calculation_tmp);
	cx_mat function_building_exponential_factor(vec excitonic_momentum,int minus);
	///rho_{n1,n2,k1-p,k2-q}(excitonic_momentum,G)=\bra{n1k1-p}e^{i(excitonic_momentum+G)r\ket{n2k2-q}
	tuple<cx_mat,cx_mat> pull_values(vec excitonic_momentum,vec p_parameter,vec q_parameter,int diagonal_k,int opp_conjungtion,int minus);
	cx_mat pull_reduced_values_vc(cx_mat rho,int diagonal_k,int reverse);
	cx_mat pull_reduced_values_cv(cx_mat rho,int diagonal_k,int reverse);
	cx_mat pull_reduced_values_cc_vv(cx_mat rho,int conduction_or_valence, int reverse,int diagonal_k);
	////term 0: total, term 1: vc, term 2: cc term 3: vv
	void print(vec excitonic_momentum,vec q_parameter,int which_term,int diagonal_k,int minus);
};
Dipole_Elements::Dipole_Elements(int number_k_points_list_tmp, mat k_points_list_tmp, int number_g_points_list_tmp, mat g_points_list_tmp, int number_wannier_centers_tmp, int number_valence_bands_tmp, int number_conduction_bands_tmp, Hamiltonian_TB *hamiltonian_tb_tmp,int spinorial_calculation_tmp){
	number_g_points_list=number_g_points_list_tmp;
	number_conduction_bands=number_conduction_bands_tmp;
	number_valence_bands=number_valence_bands_tmp;
	number_k_points_list=number_k_points_list_tmp;
	number_wannier_centers=number_wannier_centers_tmp;
	number_valence_plus_conduction=number_valence_bands+number_conduction_bands;
	k_points_list=k_points_list_tmp;
	g_points_list=g_points_list_tmp;
	hamiltonian_tb=hamiltonian_tb_tmp;
	spinorial_calculation=spinorial_calculation_tmp;
	htb_basis_dimension=hamiltonian_tb->pull_htb_basis_dimension();
	spin_htb_basis_dimension=htb_basis_dimension/(spinorial_calculation+1);
	spin_number_valence_plus_conduction=(spinorial_calculation+1)*number_valence_plus_conduction;
	spin_number_valence_bands=(spinorial_calculation+1)*number_valence_bands;
	spin_number_conduction_bands=(spinorial_calculation+1)*number_conduction_bands;
	spin_number_valence_bands=(spinorial_calculation+1)*number_valence_bands;
	if (spinorial_calculation == 1){
		wannier_centers.set_size(2);
		wannier_centers(0).set_size(3,number_wannier_centers);
		wannier_centers(1).set_size(3,number_wannier_centers);
	}else{
		wannier_centers.set_size(1);
		wannier_centers(0).set_size(3,number_wannier_centers);
	}
	wannier_centers=hamiltonian_tb->pull_wannier_centers();

};
cx_mat Dipole_Elements::function_building_exponential_factor(vec excitonic_momentum, int minus){
	cx_mat exponential_factor_tmp(htb_basis_dimension,number_g_points_list);

	#pragma omp parallel for collapse(3)
	for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
		for(int g=0; g<number_g_points_list; g++)
			for(int i=0; i<spin_htb_basis_dimension; i++){
				exponential_factor_tmp(spin_channel*spin_htb_basis_dimension+i,g).real(cos(accu((wannier_centers(spin_channel)).col(i)%((1-minus*2)*(g_points_list.col(g)+excitonic_momentum)))));
				exponential_factor_tmp(spin_channel*spin_htb_basis_dimension+i,g).imag(sin(accu((wannier_centers(spin_channel)).col(i)%((1-minus*2)*(g_points_list.col(g)+excitonic_momentum)))));
			}

	return exponential_factor_tmp;
};

///diagonal_k ---> rho_{n1,n2,k1-p,k2-q}(excitonic_momentum,G)-->rho_{n1,n2,k1-p,k1-q}(excitonic_momentum,G)
tuple<cx_mat,cx_mat> Dipole_Elements::pull_values(vec excitonic_momentum,vec p_parameter,vec q_parameter,int diagonal_k,int opp_conjungtion, int minus){
	///adding exponential term e^{i(k+G)r} to the right states
	///e_{gl}k_{gm} -> l_{g(l,m)}
	///calculating the exponenential factor at the beginning
	if(opp_conjungtion==1)
		minus=opp_conjungtion;

	cx_mat exponential_factor=function_building_exponential_factor(excitonic_momentum,minus);
	
	int effective_number_k_points_list;
	if(diagonal_k==1)
		effective_number_k_points_list=number_k_points_list;
	else
		effective_number_k_points_list=number_k_points_list*number_k_points_list;
	
	cx_mat energies((spinorial_calculation+1),number_conduction_bands*number_valence_bands*number_k_points_list);
	cx_mat rho(spin_number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list,number_g_points_list);
	///in order to avoid twice the calculations in the case of diagonal_k=0, the two possibilities have been separated
	////this is the heaviest but also the fastest solution (to use cx_cub for left state instead of cx_mat)
	if(diagonal_k==1){
		////initializing memory
		cx_mat energies_tot((spinorial_calculation+1),number_k_points_list*number_valence_plus_conduction*number_valence_plus_conduction);
		cx_mat ks_state(htb_basis_dimension, number_valence_plus_conduction); cx_mat ks_state_q(htb_basis_dimension, number_valence_plus_conduction); 
		mat ks_energy(2,number_valence_plus_conduction); mat ks_energy_q(2,number_valence_plus_conduction);
		tuple<mat,cx_mat> ks_states_k_point(ks_energy,ks_state); tuple<mat,cx_mat> ks_states_k_point_q(ks_energy_q,ks_state_q);
		
		cx_cube ks_state_right(htb_basis_dimension,number_valence_plus_conduction*effective_number_k_points_list,number_g_points_list);
		cx_cube ks_state_left(htb_basis_dimension,number_valence_plus_conduction*effective_number_k_points_list,number_g_points_list);
		
		#pragma omp parallel for private(ks_states_k_point,ks_states_k_point_q,ks_state,ks_state_q,ks_energy,ks_energy_q)
		for(int i=0;i<number_k_points_list;i++){
				ks_states_k_point = hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i)-p_parameter,number_valence_bands,number_conduction_bands);
				ks_states_k_point_q = hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i)-q_parameter,number_valence_bands,number_conduction_bands);
				ks_state=get<1>(ks_states_k_point); ks_state_q=get<1>(ks_states_k_point_q);
				ks_energy=get<0>(ks_states_k_point); ks_energy_q=get<0>(ks_states_k_point_q);
				for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++){
					for(int m=0;m<number_valence_plus_conduction;m++){
						for(int g=0;g<number_g_points_list;g++){
							ks_state_right.subcube(spin_channel*spin_htb_basis_dimension,m*number_k_points_list+i,g,(spin_channel+1)*spin_htb_basis_dimension-1,m*number_k_points_list+i,g)=
								exponential_factor.submat(spin_channel*spin_htb_basis_dimension,g,(spin_channel+1)*spin_htb_basis_dimension-1,g)%ks_state_q.submat(spin_channel*spin_htb_basis_dimension,m,(spin_channel+1)*spin_htb_basis_dimension-1,m);
							ks_state_left.subcube(spin_channel*spin_htb_basis_dimension,m*number_k_points_list+i,g,(spin_channel+1)*spin_htb_basis_dimension-1,m*number_k_points_list+i,g)=
								ks_state.submat(spin_channel*spin_htb_basis_dimension,m,(spin_channel+1)*spin_htb_basis_dimension-1,m);
						}
						for(int n=0;n<number_valence_plus_conduction;n++)
							energies_tot(spin_channel,m*number_valence_plus_conduction*number_k_points_list+n*number_k_points_list+i).real(ks_energy(spin_channel,m)-ks_energy_q(spin_channel,n));
					}
				}
		}
		for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++){
			#pragma omp parallel for collapse(2) private(spin_channel)
				for(int m=0;m<number_valence_plus_conduction;m++)
					for(int n=0;n<number_valence_plus_conduction;n++)
						rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*number_k_points_list+m*number_valence_plus_conduction*number_k_points_list+n*number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*number_k_points_list+m*number_valence_plus_conduction*number_k_points_list+n*number_k_points_list+number_k_points_list-1,number_g_points_list-1)=
							(cx_mat)(sum(conj(ks_state_left.subcube(spin_channel*spin_htb_basis_dimension,m*number_k_points_list,0,(spin_channel+1)*spin_htb_basis_dimension-1,(m+1)*number_k_points_list-1,number_g_points_list-1))%
							ks_state_right.subcube(spin_channel*spin_htb_basis_dimension,n*number_k_points_list,0,(spin_channel+1)*spin_htb_basis_dimension-1,(n+1)*number_k_points_list-1,number_g_points_list-1),0));
		}
		///extracting energy differences Conduction - Valence
		for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
			for(int v=0;v<number_valence_bands;v++)
				for(int c=0;c<number_conduction_bands;c++)
					for(int k=0;k<number_k_points_list;k++)
						energies(spin_channel,c*number_valence_bands*number_k_points_list+v*number_k_points_list+k)=energies_tot(spin_channel,v*number_valence_plus_conduction*number_k_points_list+(number_valence_bands+c)*number_k_points_list+k);
		
		ks_state_right.reset();
		ks_state_left.reset();
		ks_state.reset();
		ks_energy.reset();

	}else{
		cx_mat energies((spinorial_calculation+1),effective_number_k_points_list*number_valence_plus_conduction*number_valence_plus_conduction,fill::zeros);
		cx_cube ks_states_k_points(htb_basis_dimension,number_valence_plus_conduction,number_k_points_list);
		cx_cube ks_states_k_points_q(htb_basis_dimension,number_valence_plus_conduction,number_k_points_list);

		///SCRIVERLO IN UN MODO PIU SMART !!!!
		cx_cube ks_state_right(htb_basis_dimension,effective_number_k_points_list,number_g_points_list);
		cx_cube ks_state_left(htb_basis_dimension,effective_number_k_points_list,number_g_points_list);

		cout<<"diagonalization"<<endl;
		for(int i=0;i<number_k_points_list;i++){
			ks_states_k_points.slice(i) = get<1>(hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i)-p_parameter,number_valence_bands,number_conduction_bands));
			ks_states_k_points_q.slice(i) = get<1>(hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i)-q_parameter,number_valence_bands,number_conduction_bands));
		}
		cout<<"combination"<<endl;
		cx_mat ks_state_q(htb_basis_dimension,number_valence_plus_conduction);
		#pragma omp parallel for collapse(3) shared(rho,ks_state_right,ks_state_left,exponential_factor,ks_state_q)
		for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
			for(int m=0;m<number_valence_plus_conduction;m++)
				for(int n=0;n<number_valence_plus_conduction;n++){
					for(int j=0;j<number_k_points_list;j++){
						ks_state_q=ks_states_k_points_q.slice(j);
						for(int i=0;i<number_k_points_list;i++)
							for(int g=0;g<number_g_points_list;g++){
								ks_state_right.subcube(spin_channel*spin_htb_basis_dimension,i*number_k_points_list+j,g,(spin_channel+1)*spin_htb_basis_dimension-1,i*number_k_points_list+j,g)=
									exponential_factor.submat(spin_channel*spin_htb_basis_dimension,g,(spin_channel+1)*spin_htb_basis_dimension-1,g)%ks_state_q.submat(spin_channel*spin_htb_basis_dimension,n,(spin_channel+1)*spin_htb_basis_dimension-1,n);
								ks_state_left.subcube(spin_channel*spin_htb_basis_dimension,i*number_k_points_list+j,g,(spin_channel+1)*spin_htb_basis_dimension-1,i*number_k_points_list+j,g)=
									ks_states_k_points.subcube(spin_channel*spin_htb_basis_dimension,m,i,(spin_channel+1)*spin_htb_basis_dimension-1,m,i);
							}
					}
					if(opp_conjungtion==0){
						rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+m*number_valence_plus_conduction*effective_number_k_points_list+n*effective_number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+m*number_valence_plus_conduction*effective_number_k_points_list+n*effective_number_k_points_list+effective_number_k_points_list-1,number_g_points_list-1)=
							(cx_mat)(sum(conj(ks_state_left.subcube(spin_channel*spin_htb_basis_dimension,0,0,(spin_channel+1)*spin_htb_basis_dimension-1,effective_number_k_points_list-1,number_g_points_list-1))%
							ks_state_right.subcube(spin_channel*spin_htb_basis_dimension,0,0,(spin_channel+1)*spin_htb_basis_dimension-1,effective_number_k_points_list-1,number_g_points_list-1),0));
					}else{
						rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+m*number_valence_plus_conduction*effective_number_k_points_list+n*effective_number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+m*number_valence_plus_conduction*effective_number_k_points_list+n*effective_number_k_points_list+effective_number_k_points_list-1,number_g_points_list-1)=
							(cx_mat)(sum(ks_state_left.subcube(spin_channel*spin_htb_basis_dimension,0,0,(spin_channel+1)*spin_htb_basis_dimension-1,effective_number_k_points_list-1,number_g_points_list-1)%
							conj(ks_state_right.subcube(spin_channel*spin_htb_basis_dimension,0,0,(spin_channel+1)*spin_htb_basis_dimension-1,effective_number_k_points_list-1,number_g_points_list-1)),0));
					}
				}
		ks_state_right.reset();
		ks_state_left.reset();
		ks_states_k_points.reset();
		ks_states_k_points_q.reset();
	}

	return {energies,rho};
};
cx_mat Dipole_Elements::pull_reduced_values_vc(cx_mat rho, int diagonal_k, int reverse){
	int effective_number_k_points_list;
	if(diagonal_k==0)
		effective_number_k_points_list=number_k_points_list*number_k_points_list;
	else
		effective_number_k_points_list=number_k_points_list;

	spin_number_valence_bands=(spinorial_calculation+1)*number_valence_bands;
	cx_mat rho_reduced(spin_number_valence_bands*number_conduction_bands*effective_number_k_points_list,number_g_points_list);
	//rho_reduced = rho(vc)
	if(reverse==1){
		#pragma omp parallel for collapse(3)
		for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
			for(int m=0;m<number_conduction_bands;m++)
				for(int n=0;n<number_valence_bands;n++)
					rho_reduced.submat(spin_channel*number_valence_bands*number_conduction_bands*effective_number_k_points_list+m*number_valence_bands*effective_number_k_points_list+n*effective_number_k_points_list,0,spin_channel*number_valence_bands*number_conduction_bands*effective_number_k_points_list+m*number_valence_bands*effective_number_k_points_list+(n+1)*effective_number_k_points_list-1,number_g_points_list-1)=
						rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+n*number_valence_plus_conduction*effective_number_k_points_list+(number_valence_bands+m)*effective_number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+n*number_valence_plus_conduction*effective_number_k_points_list+(number_valence_bands+m+1)*effective_number_k_points_list-1,number_g_points_list-1);
	}else{
		#pragma omp parallel for collapse(3)
		for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
			for(int m=0;m<number_conduction_bands;m++)
				for(int n=0;n<number_valence_bands;n++)
					rho_reduced.submat(spin_channel*number_valence_bands*number_conduction_bands*effective_number_k_points_list+n*number_conduction_bands*effective_number_k_points_list+m*effective_number_k_points_list,0,spin_channel*number_valence_bands*number_conduction_bands*effective_number_k_points_list+n*number_conduction_bands*effective_number_k_points_list+(m+1)*effective_number_k_points_list-1,number_g_points_list-1)=
						rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+n*number_valence_plus_conduction*effective_number_k_points_list+(number_valence_bands+m)*effective_number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+n*number_valence_plus_conduction*effective_number_k_points_list+(number_valence_bands+m+1)*effective_number_k_points_list-1,number_g_points_list-1);
	}
	return rho_reduced;
};
cx_mat Dipole_Elements::pull_reduced_values_cv(cx_mat rho, int diagonal_k, int reverse){
	int effective_number_k_points_list;
	if(diagonal_k==0)
		effective_number_k_points_list=number_k_points_list*number_k_points_list;
	else
		effective_number_k_points_list=number_k_points_list;

	spin_number_valence_bands=(spinorial_calculation+1)*number_valence_bands;
	cx_mat rho_reduced (spin_number_valence_bands*number_conduction_bands*effective_number_k_points_list,number_g_points_list);
	//rho_reduced = rho(vc)
	///cout<<"dimension: "<<size(rho)<<endl;
	if(reverse==0){
		#pragma omp parallel for collapse(2)
		for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
			for(int m=0;m<number_conduction_bands;m++)
				rho_reduced.submat(spin_channel*number_conduction_bands*number_valence_bands*effective_number_k_points_list+m*number_valence_bands*effective_number_k_points_list,0,spin_channel*number_conduction_bands*number_valence_bands*effective_number_k_points_list+(m+1)*number_valence_bands*effective_number_k_points_list-1,number_g_points_list-1)=
					rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+(m+number_valence_bands)*number_valence_plus_conduction*effective_number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+(m+number_valence_bands)*number_valence_plus_conduction*effective_number_k_points_list+number_valence_bands*effective_number_k_points_list-1,number_g_points_list-1);
	}else{
		#pragma omp parallel for collapse(2)
		for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
			for(int m=0;m<number_conduction_bands;m++)
				for(int n=0;n<number_valence_bands;n++)
					rho_reduced.submat(spin_channel*number_valence_bands*number_conduction_bands*effective_number_k_points_list+n*number_conduction_bands*effective_number_k_points_list+m*effective_number_k_points_list,0,spin_channel*number_valence_bands*number_conduction_bands*effective_number_k_points_list+n*number_conduction_bands*effective_number_k_points_list+(m+1)*effective_number_k_points_list,number_g_points_list-1)=
						rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+(m+number_valence_bands)*number_valence_plus_conduction*effective_number_k_points_list+n*effective_number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+(m+number_valence_bands)*number_valence_plus_conduction*effective_number_k_points_list+(n+1)*effective_number_k_points_list-1,number_g_points_list-1);
	}
	return rho_reduced;
};
cx_mat Dipole_Elements:: pull_reduced_values_cc_vv(cx_mat rho, int conduction_or_valence,int reverse,int diagonal_k){
	int effective_number_k_points_list;
	if(diagonal_k==0)
		effective_number_k_points_list=number_k_points_list*number_k_points_list;
	else
		effective_number_k_points_list=number_k_points_list;
	if(conduction_or_valence==0){
		spin_number_conduction_bands=(spinorial_calculation+1)*number_conduction_bands;
		cx_mat rho_reduced(spin_number_conduction_bands*number_conduction_bands*effective_number_k_points_list,number_g_points_list);
		//rho_reducde = rho(cc)
		if(reverse==0){
			for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
				for(int m=0;m<number_conduction_bands;m++)
					rho_reduced.submat(spin_channel*number_conduction_bands*number_conduction_bands*effective_number_k_points_list+m*number_conduction_bands*effective_number_k_points_list,0,spin_channel*number_conduction_bands*number_conduction_bands*effective_number_k_points_list+m*number_conduction_bands*effective_number_k_points_list+number_conduction_bands*effective_number_k_points_list-1,number_g_points_list-1)=
						rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+(number_valence_bands+m)*number_valence_plus_conduction*effective_number_k_points_list+number_valence_bands*effective_number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+(number_valence_bands+m)*number_valence_plus_conduction*effective_number_k_points_list+number_valence_plus_conduction*effective_number_k_points_list-1,number_g_points_list-1);
			return rho_reduced;
		}else{
			for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
				for(int m=0;m<number_conduction_bands;m++)
					for(int n=0;n<number_conduction_bands;n++)
						rho_reduced.submat(spin_channel*number_conduction_bands*number_conduction_bands*effective_number_k_points_list+m*number_conduction_bands*effective_number_k_points_list+n*effective_number_k_points_list,0,spin_channel*number_conduction_bands*number_conduction_bands*effective_number_k_points_list+m*number_conduction_bands*effective_number_k_points_list+(n+1)*effective_number_k_points_list-1,number_g_points_list-1)=
							rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+(number_valence_bands+n)*number_valence_plus_conduction*effective_number_k_points_list+(m+number_valence_bands)*effective_number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+(number_valence_bands+n)*number_valence_plus_conduction*effective_number_k_points_list+(number_valence_bands+m+1)*effective_number_k_points_list-1,number_g_points_list-1);
			return rho_reduced;
		}
	}else{
		spin_number_valence_bands=(spinorial_calculation+1)*number_valence_bands;
		cx_mat rho_reduced(spin_number_valence_bands*number_valence_bands*effective_number_k_points_list,number_g_points_list);
		//rho_reducde = rho(vv)
		if(reverse==0){
			for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
				for(int m=0;m<number_valence_bands;m++)
					rho_reduced.submat(spin_channel*number_valence_bands*number_valence_bands*effective_number_k_points_list+m*number_valence_bands*effective_number_k_points_list,0,spin_channel*number_valence_bands*number_valence_bands*effective_number_k_points_list+m*number_valence_bands*effective_number_k_points_list+number_valence_bands*effective_number_k_points_list-1,number_g_points_list-1)=
						rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+m*number_valence_plus_conduction*effective_number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+m*number_valence_plus_conduction*effective_number_k_points_list+number_valence_bands*effective_number_k_points_list-1,number_g_points_list-1);
			return rho_reduced;
		}else{
			for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
				for(int m=0;m<number_valence_bands;m++)
					for(int n=0;n<number_valence_bands;n++)
						rho_reduced.submat(spin_channel*number_valence_bands*number_valence_bands*effective_number_k_points_list+m*number_valence_bands*effective_number_k_points_list+n*effective_number_k_points_list,0,spin_channel*number_valence_bands*number_valence_bands*effective_number_k_points_list+m*number_valence_bands*effective_number_k_points_list+(n+1)*effective_number_k_points_list-1,number_g_points_list-1)=
							rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+n*number_valence_plus_conduction*effective_number_k_points_list+m*effective_number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list+n*number_valence_plus_conduction*effective_number_k_points_list+(m+1)*effective_number_k_points_list-1,number_g_points_list-1);
			return rho_reduced;
		}
	}
};
void Dipole_Elements::print(vec excitonic_momentum, vec p_parameter, int which_term, int diagonal_k, int minus){
	cx_mat exponential_factor=function_building_exponential_factor(excitonic_momentum,minus);
	
	int effective_number_k_points_list;
	if(diagonal_k==0)
		effective_number_k_points_list=number_k_points_list*number_k_points_list;
	else
		effective_number_k_points_list=number_k_points_list;

	//for(int g=0;g<number_g_points_list;g++)
	//	cout<<exponential_factor.col(g)<<" ";
	///for(int i=0;i<number_k_points_list;i++)
	///	cout<<k_points_list.col(i)<<" ";
	tuple<cx_mat,cx_mat> energies_and_dipole_elements=pull_values(excitonic_momentum,excitonic_momentum,p_parameter,diagonal_k,0,0);
	cx_mat energies=get<0>(energies_and_dipole_elements);
	cx_mat dipole_elements_tot=get<1>(energies_and_dipole_elements);
	int states1;
	if(which_term==0){
		states1=spin_number_valence_plus_conduction*number_valence_plus_conduction*effective_number_k_points_list;
		for(int i=0;i<states1;i++){
			for(int g=0;g<number_g_points_list;g++)
				cout<<dipole_elements_tot(i,g)<<" ";
			cout<<endl;
		}
	}else if(which_term==1){
		states1=spin_number_valence_bands*number_conduction_bands*effective_number_k_points_list;
		cx_mat dipole_elements_partial=pull_reduced_values_vc(dipole_elements_tot,diagonal_k,0);
		for(int i=0;i<states1;i++){
			for(int g=0;g<number_g_points_list;g++)
				cout<<dipole_elements_partial(i,g)<<" ";
			cout<<endl;
		}
	}else if(which_term==2){
		states1=spin_number_conduction_bands*number_conduction_bands*effective_number_k_points_list;
		cx_mat dipole_elements_partial=pull_reduced_values_cc_vv(dipole_elements_tot,0,0,diagonal_k);
		int count=0;
		for(int spin=0;spin<1;spin++)
			for(int c1=0;c1<number_conduction_bands;c1++)
				for(int c2=0;c2<number_conduction_bands;c2++)
					for(int k=0;k<effective_number_k_points_list;k++){
						cout<<spin<<" "<<c1<<" "<<c2<<" "<<k<<endl;
						for(int g=0;g<number_g_points_list;g++)
							cout<<dipole_elements_partial(spin*number_conduction_bands*number_conduction_bands*effective_number_k_points_list+c1*number_conduction_bands*effective_number_k_points_list+c2*effective_number_k_points_list+k,g)<<" ";
						cout<<endl;
					}
	}else if(which_term==4){
		states1=spin_number_valence_bands*number_conduction_bands*effective_number_k_points_list;
		cx_mat dipole_elements_partial=pull_reduced_values_cv(dipole_elements_tot,diagonal_k,0);
		for(int i=0;i<states1;i++){
			for(int g=0;g<number_g_points_list;g++)
				cout<<dipole_elements_partial(i,g)<<" ";
			cout<<endl;
		}
	}else{
		states1=spin_number_valence_bands*number_valence_bands*effective_number_k_points_list;
		cx_mat dipole_elements_partial=pull_reduced_values_cc_vv(dipole_elements_tot,1,0,diagonal_k);
		for(int i=0;i<states1;i++){
			for(int g=0;g<number_g_points_list;g++)
				cout<<dipole_elements_partial(i,g)<<" ";
			cout<<endl;
		}
	}
};
/// Dielectric_Function
class Dielectric_Function
{
private:
	int number_k_points_list;
	int number_g_points_list;
	mat g_points_list;
	int number_valence_bands;
	int number_conduction_bands;
	vec excitonic_momentum;
	Dipole_Elements *dipole_elements;
	Coulomb_Potential *coulomb_potential;
	int spinorial_calculation;
	cx_mat rho_reduced;
	mat energies;
public:
	Dielectric_Function(Dipole_Elements *dipole_elements_tmp,int number_k_points_list_tmp,int number_g_points_list_tmp,mat g_points_list_tmp,int number_valence_bands_tmp,int number_conduction_bands_tmp,Coulomb_Potential *coulomb_potential_tmp,int spinorial_calculation_tmp);
	cx_mat pull_values(vec excitonic_momentum,cx_double omega,double eta,int order_approximation);
	cx_mat pull_values_PPA(vec excitonic_momentum,cx_double omega,double eta,double PPA,int order_approximation);
	void print(vec excitonic_momentum,cx_double omega,double eta,double PPA,int which_term,int order_approximation);
	void pull_macroscopic_value(cx_vec omegas_path,int number_omegas_path,double eta,string file_macroscopic_dielectric_function_name,int order_approximation);
	~Dielectric_Function(){
		coulomb_potential=NULL;
		dipole_elements=NULL;
	};
};
Dielectric_Function::Dielectric_Function(Dipole_Elements *dipole_elements_tmp,int number_k_points_list_tmp,int number_g_points_list_tmp,mat g_points_list_tmp,int number_valence_bands_tmp,int number_conduction_bands_tmp,Coulomb_Potential *coulomb_potential_tmp,int spinorial_calculation_tmp){
	number_conduction_bands=number_conduction_bands_tmp;
	number_valence_bands=number_valence_bands_tmp;
	number_k_points_list=number_k_points_list_tmp;
	number_g_points_list=number_g_points_list_tmp;
	dipole_elements=dipole_elements_tmp;
	coulomb_potential=coulomb_potential_tmp;
	g_points_list=g_points_list_tmp;
	spinorial_calculation=spinorial_calculation_tmp;
};
cx_mat Dielectric_Function::pull_values(vec excitonic_momentum, cx_double omega, double eta, int order_approximation){
	cx_mat epsiloninv(number_g_points_list,number_g_points_list,fill::zeros);
	cx_double ieta; ieta.real(0.0); ieta.imag(eta);
	int number_valence_plus_conduction=number_conduction_bands+number_valence_bands;

	vec zeros_vec(3,fill::zeros);
	tuple<cx_mat,cx_mat> energies_rho=dipole_elements->pull_values(excitonic_momentum,zeros_vec,excitonic_momentum,1,0,0);
	cx_mat rho_cv=dipole_elements->pull_reduced_values_cv(get<1>(energies_rho),1,0);
	cx_mat energies=get<0>(energies_rho);
	cx_vec coulomb_shifted(number_g_points_list);
	cx_double ione; ione.real(1.0); ione.imag(0.0);
	
	//auto t1 = std::chrono::high_resolution_clock::now();
	/// defining the denominator factors
	cx_mat rho_reduced_single_column_modified((spinorial_calculation+1)*number_k_points_list*number_conduction_bands*number_valence_bands,number_g_points_list);
	cx_vec multiplicative_factor((spinorial_calculation+1)*number_k_points_list*number_conduction_bands*number_valence_bands);
	for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++){
		#pragma omp parallel for collapse(3) 
		for(int i=0;i<number_k_points_list;i++)
			for(int c=0;c<number_conduction_bands;c++)
				for(int v=0;v<number_valence_bands;v++){
					multiplicative_factor(spin_channel*number_conduction_bands*number_valence_bands*number_k_points_list+c*number_valence_bands*number_k_points_list+v*number_k_points_list+i)= ione / (omega + energies(spin_channel,c*number_valence_bands*number_k_points_list+v*number_k_points_list+i) + ieta) -  ione / (omega - energies(spin_channel,c*number_valence_bands*number_k_points_list+v*number_k_points_list+i) - ieta);
					//cout<<multiplicative_factor(spin_channel*number_conduction_bands*number_valence_bands*number_k_points_list+c*number_valence_bands*number_k_points_list+v*number_k_points_list+i)<<endl;
				}
	}
	#pragma omp parallel for
	for(int i=0;i<number_g_points_list;i++){
		coulomb_shifted(i).real(coulomb_potential->pull(excitonic_momentum+g_points_list.col(i)));
		rho_reduced_single_column_modified.col(i)=rho_cv.col(i)%multiplicative_factor;
	}
	if(order_approximation==0){
		#pragma omp parallel for collapse(2) shared(epsiloninv)
		for(int i=0;i<number_g_points_list;i++)
			for(int j=0;j<number_g_points_list;j++)
				epsiloninv(i,j)=coulomb_shifted(i)*accu(conj(rho_cv.col(i))%rho_reduced_single_column_modified.col(j));

		#pragma omp parallel for 
		for(int i=0;i<number_g_points_list;i++)
			epsiloninv(i,i).real(epsiloninv(i,i).real()+1.0);
	}else{
		cx_mat epsiloninv_tmp1; epsiloninv_tmp1.zeros(number_g_points_list,number_g_points_list);
		cx_mat epsiloninv_tmp2; epsiloninv_tmp2.zeros(number_g_points_list,number_g_points_list);

		#pragma omp parallel for collapse(2)
		for(int i=0;i<number_g_points_list;i++)
			for(int j=0;j<number_g_points_list;j++){
				epsiloninv_tmp1(i,j)=coulomb_shifted(i)*accu(conj(rho_cv.col(j))%rho_reduced_single_column_modified.col(i));
				epsiloninv_tmp2(i,j)=epsiloninv_tmp1(i,j);
			}

		///RPA approximation solving Dyson equation
		#pragma omp parallel for 
		for(int i=0;i<number_g_points_list;i++)
			epsiloninv_tmp1(i,i).real(1.0-epsiloninv_tmp1(i,i).real());
		
		epsiloninv=epsiloninv_tmp1.i()*epsiloninv_tmp2;
		#pragma omp parallel for 
		for(int i=0;i<number_g_points_list;i++)
			epsiloninv(i,i).real(epsiloninv(i,i).real()+1.0);
	
	epsiloninv_tmp1.reset();
	epsiloninv_tmp2.reset();
	}
	
	multiplicative_factor.reset();
	rho_reduced_single_column_modified.reset();

	return epsiloninv;
};
cx_mat Dielectric_Function::pull_values_PPA(vec excitonic_momentum,cx_double omega,double eta,double PPA,int order_approximation){
	cx_double omega_PPA; omega_PPA.imag(PPA); omega_PPA.real(0.0);
	cx_double omega_0; omega_0.real(0.0); omega_0.imag(0.0);
	cx_double ieta; ieta.real(0.0); ieta.imag(eta);
	
	cx_mat epsiloninv_0=pull_values(excitonic_momentum,omega_0,eta,order_approximation);
	cx_mat epsiloninv_PPA=pull_values(excitonic_momentum,omega_PPA,eta,order_approximation);
	
	cx_mat rgg(number_g_points_list,number_g_points_list);
	cx_mat ogg(number_g_points_list,number_g_points_list);
	
	ogg=PPA*sqrt(epsiloninv_PPA/(epsiloninv_0-epsiloninv_PPA));
	rgg=(epsiloninv_0%ogg)/2;
	
	cx_mat epsilon_app(number_g_points_list,number_g_points_list,fill::zeros);

	for(int i=0;i<number_g_points_list;i++)
		for(int j=0;j<number_g_points_list;j++)
			epsilon_app(i,j)=rgg(i,j)*(1.0/(omega-ogg(i,j)+ieta)-1.0/(omega+ogg(i,j)-ieta));
	for(int i=0;i<number_g_points_list;i++)
		epsilon_app(i,i)+=1.0;
	return epsilon_app;
};
void Dielectric_Function::pull_macroscopic_value(cx_vec omegas_path,int number_omegas_path,double eta,string file_macroscopic_dielectric_function_name,int order_approximation){
	cx_mat macroscopic_dielectric_function(number_g_points_list,number_g_points_list);
	vec q_point_0(3,fill::zeros);
	cx_double ieta; ieta.real(0.0); ieta.imag(eta);
	q_point_0(0)+=minval;
	int g_point_0=int(number_g_points_list/2);
	ofstream file_macroscopic_dielectric_function;
	file_macroscopic_dielectric_function.open(file_macroscopic_dielectric_function_name);
	for(int i=0;i<number_omegas_path;i++){
		macroscopic_dielectric_function=(pull_values(q_point_0,omegas_path(i),eta,order_approximation)).i();
		macroscopic_dielectric_function(g_point_0,g_point_0).imag(macroscopic_dielectric_function(g_point_0,g_point_0).imag()*1/pow(minval,2));
		cout<<i<<" "<<number_omegas_path<<-macroscopic_dielectric_function(g_point_0,g_point_0)<<endl;
		file_macroscopic_dielectric_function<<i<<" "<<omegas_path(i)<<" "<<-macroscopic_dielectric_function(g_point_0,g_point_0)<<endl;
	}
	file_macroscopic_dielectric_function.close();
};
void Dielectric_Function::print(vec excitonic_momentum,cx_double omega,double eta,double PPA,int which_term,int order_approximation){
	cx_mat dielectric_function(number_g_points_list,number_g_points_list);
	if(which_term==0)
		dielectric_function=pull_values(excitonic_momentum,omega,eta,order_approximation);
	else
		dielectric_function=pull_values_PPA(excitonic_momentum,omega,eta,PPA,order_approximation);
	
	for(int i=0;i<number_g_points_list;i++){
		for(int j=0;j<number_g_points_list;j++)
			cout<<dielectric_function(i,j)<<" ";	
		cout<<endl;
	}
};	
/// Excitonic_Hamiltonian class
class Excitonic_Hamiltonian
{
private:
	int spinorial_calculation;
	int number_valence_bands;
	int number_conduction_bands;
	int number_valence_plus_conduction;
	int dimension_bse_hamiltonian;
	int spin_dimension_bse_hamiltonian;
	int spin_dimension_bse_hamiltonian_2;
	int spin_number_valence_plus_conduction;
	int htb_basis_dimension;
	int number_k_points_list;
	int number_valence_times_conduction;
	int number_g_points_list;
	double cell_volume;
	mat k_points_list;
	mat g_points_list;
	mat exciton;
	mat exciton_spin;
	Dipole_Elements *dipole_elements;
	cx_mat v_coulomb_gg;
	cx_vec v_coulomb_g;
	cx_mat excitonic_hamiltonian;
	cx_mat rho_q_diagk_cv;
	vec excitonic_momentum{vec(3)};
public:
	/// be carefull: do not try to build the BSE matrix with more bands than those given by the hamiltonian!!!
	/// there is a check at the TB hamiltonian level but not here...
	Excitonic_Hamiltonian(int number_valence_bands_tmp,int number_conduction_bands_tmp, mat k_points_list_tmp, int number_k_points_list_tmp, mat g_points_list_tmp,int number_g_points_list_tmp, int spinorial_calculation_tmp, int htb_basis_dimension_tmp,Dipole_Elements *dipole_elements_tmp, double cell_volume_tmp);
	void pull_coulomb_potentials(Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,vec excitonic_momentum,double eta,int order_approximation);
	void pull_resonant_part_and_rcv(vec excitonic_momentum_tmp,double eta);
	void add_coupling_part();
	tuple<cx_mat,cx_mat> extract_hbse_and_rcv(vec excitonic_momentum_tmp,double eta,Coulomb_Potential *coulomb_potential,Dielectric_Function *dielectric_function,int adding_screening,int tamn_dancoff,int order_approximation);
	tuple<cx_vec,cx_mat> common_diagonalization(vec excitonic_momentum_tmp,double eta,Coulomb_Potential *coulomb_potential,Dielectric_Function *dielectric_function,int adding_screening,int tamn_dancoff,int order_approximation);
	tuple<vec,cx_mat> cholesky_diagonalization(vec excitonic_momentum_tmp,double eta,Coulomb_Potential *coulomb_potential,Dielectric_Function *dielectric_function,int adding_screenin,int order_approximation);
	cx_vec pull_excitonic_oscillator_force(cx_mat excitonic_eigenstates);
	void pull_macroscopic_bse_dielectric_function(cx_vec omegas_path,int number_omegas_path,double eta,string file_macroscopic_dielectric_function_bse_name,double lorentzian,int tamn_dancoff,Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,int order_approximation);
	void print(vec excitonic_momentum_tmp,double eta,int tamn_dancoff,Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,int order_approximation);
};
Excitonic_Hamiltonian::Excitonic_Hamiltonian(int number_valence_bands_tmp,int number_conduction_bands_tmp, mat k_points_list_tmp, int number_k_points_list_tmp, mat g_points_list_tmp,int number_g_points_list_tmp,int spinorial_calculation_tmp,int htb_basis_dimension_tmp,Dipole_Elements *dipole_elements_tmp,double cell_volume_tmp){
	spinorial_calculation = spinorial_calculation_tmp;
	number_k_points_list = number_k_points_list_tmp;
	number_conduction_bands = number_conduction_bands_tmp;
	number_valence_bands = number_valence_bands_tmp;
	number_valence_times_conduction = number_conduction_bands*number_valence_bands;
	number_valence_plus_conduction = number_conduction_bands + number_valence_bands;
	dimension_bse_hamiltonian = number_k_points_list * number_conduction_bands * number_valence_bands;
	k_points_list = k_points_list_tmp;
	g_points_list = g_points_list_tmp;
	number_g_points_list = number_g_points_list_tmp;
	htb_basis_dimension = htb_basis_dimension_tmp;

	cell_volume=cell_volume_tmp;
	dipole_elements=dipole_elements_tmp;

	v_coulomb_gg.set_size(number_g_points_list,number_g_points_list);
	v_coulomb_g.set_size(number_g_points_list);

	exciton.set_size(2, number_valence_bands*number_conduction_bands);
	int e = 0;
	for (int v = 0; v < number_valence_bands; v++)
		for (int c = 0; c < number_conduction_bands; c++){
			exciton(0, e) = c;
			exciton(1, e) = v;
			e++;
		}

	/// defining the possible spin combinations
	exciton_spin.zeros(2, 4);
	exciton_spin(1, 1) = 1;
	exciton_spin(0, 2) = 1;
	exciton_spin(0, 3) = 1;
	exciton_spin(1, 3) = 1;
	
	if (spinorial_calculation == 1){
		spin_dimension_bse_hamiltonian=dimension_bse_hamiltonian*4;
		spin_number_valence_plus_conduction=number_valence_plus_conduction*2;
	}else{
		spin_dimension_bse_hamiltonian=dimension_bse_hamiltonian;
		spin_number_valence_plus_conduction=number_valence_plus_conduction;
	}
	spin_dimension_bse_hamiltonian_2=2*spin_dimension_bse_hamiltonian;

	cout<<"Allocating HBSE memory (and rho memory)"<<endl;
	cout<<spin_dimension_bse_hamiltonian_2<<endl;
	excitonic_hamiltonian.set_size(spin_dimension_bse_hamiltonian_2,spin_dimension_bse_hamiltonian_2);
	rho_q_diagk_cv.set_size((spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list,number_g_points_list);
	cout<<"Finished allocating HBSE memory"<<endl;

};
/// calculating the potentianl before the BSE hamiltonian building
/// calculating the generalized potential (the screened one and the unscreened-one)
void Excitonic_Hamiltonian::pull_coulomb_potentials(Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,vec excitonic_momentum_tmp,double eta,int order_approximation){
	cx_mat epsilon_inv_static(number_g_points_list,number_g_points_list);
	if(adding_screening==1){
		cx_double omega_0; omega_0.real(0.0); omega_0.imag(0.0);
		epsilon_inv_static=dielectric_function->pull_values(excitonic_momentum_tmp,omega_0,eta,order_approximation);
	}else{
		epsilon_inv_static.eye();
	}
	for (int k = 0; k < number_g_points_list; k++){
		for (int s = 0; s < number_g_points_list; s++)
			v_coulomb_gg(k,s) = epsilon_inv_static(k,s)*coulomb_potential->pull(excitonic_momentum_tmp+g_points_list.col(s));
		v_coulomb_g(k) = coulomb_potential->pull(excitonic_momentum_tmp+g_points_list.col(k));
	}
};
void Excitonic_Hamiltonian::pull_resonant_part_and_rcv(vec excitonic_momentum_tmp,double eta){
	excitonic_momentum=excitonic_momentum_tmp;
	///BEOFRE BUILDING THE BSE HAMILTONIAN, BE SURE TO HAVE BUILT THE POTENTIALS (pull_coulomb_potentials)
	///building entry 00 (resonant part)
	cout<<"building 00"<<endl;
	///building v
	//cx_mat v_matrix(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian,fill::zeros);
	cout<<"building v"<<endl;
	cout<<"beginning extracting rho"<<endl;
	vec zeros_vec(3,fill::zeros);
	tuple<cx_mat,cx_mat> energies_rho_q_diagk=dipole_elements->pull_values(excitonic_momentum,zeros_vec,excitonic_momentum,1,0,0);
	rho_q_diagk_cv=dipole_elements->pull_reduced_values_cv(get<1>(energies_rho_q_diagk),1,0);
	cx_mat energies_q=get<0>(energies_rho_q_diagk);
	cout<<"ending extracting rho"<<endl;
	cx_vec zeros_long_vec((spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list,fill::zeros);
	cx_mat temporary_matrix1(number_g_points_list,(spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list);
	cx_mat rho_q_diagk_cv_tmp(number_g_points_list,(spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list);
	cx_double factor_v;
	factor_v.real((spinorial_calculation-2)/number_k_points_list);
	factor_v.imag(0.0);
	///double factor_v=(spinorial_calculation-2);
	rho_q_diagk_cv_tmp=rho_q_diagk_cv.t();
	for(int i=0;i<(spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list;i++)
		temporary_matrix1.col(i)=factor_v*(v_coulomb_g%rho_q_diagk_cv_tmp.col(i));
	////excluding G=0
	cout<<"second part"<<endl;
	temporary_matrix1.row(0)=zeros_long_vec.t();
	rho_q_diagk_cv_tmp.row(0)=zeros_long_vec.t();
	////excluding G=0 ended
	for(int i=0;i<(spinorial_calculation+1);i++)
		for(int j=0;j<(spinorial_calculation+1);j++){
			excitonic_hamiltonian.submat(i*3*number_conduction_bands*number_valence_bands*number_k_points_list,j*3*number_conduction_bands*number_valence_bands*number_k_points_list,(i*3+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1,(j*3+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1)=
				((temporary_matrix1.submat(0,i*number_conduction_bands*number_valence_bands*number_k_points_list,number_g_points_list-1,(i+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1)).t())*conj(rho_q_diagk_cv_tmp.submat(0,j*number_conduction_bands*number_valence_bands*number_k_points_list,number_g_points_list-1,(1+j)*number_conduction_bands*number_valence_bands*number_k_points_list-1));
		}
	cout<<"third part"<<endl;
	///excitonic_hamiltonian*=(spinorial_calculation-2);
	//v_matrix.submat(i*3*number_conduction_bands*number_valence_bands*number_k_points_list,j*3*number_conduction_bands*number_valence_bands*number_k_points_list,(i*3+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1,(j*3+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1)=
	//	(temporary_matrix1.submat(i*number_conduction_bands*number_valence_bands*number_k_points_list,0,(i+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1,number_g_points_list-1))*(conj(rho_q_diagk_cv_tmp.submat(j*number_conduction_bands*number_valence_bands*number_k_points_list,0,(1+j)*number_conduction_bands*number_valence_bands*number_k_points_list-1,number_g_points_list-1)).t());
	rho_q_diagk_cv_tmp.reset();
	temporary_matrix1.reset();
	zeros_long_vec.reset();
	
	///cout<<"hermiticity v: "<<accu(v_matrix-conj(v_matrix).t())<<endl;
	///building w
	cout<<"building w"<<endl;
	///cx_mat w_matrix(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian,fill::zeros);
	cout<<"beginning extracting rho"<<endl;
	cx_mat rho_kk_cc=dipole_elements->pull_reduced_values_cc_vv(get<1>(dipole_elements->pull_values(excitonic_momentum,zeros_vec,zeros_vec,0,0,0)),0,0,0);
	cx_mat rho_qq_kk_vv=dipole_elements->pull_reduced_values_cc_vv(get<1>(dipole_elements->pull_values(excitonic_momentum,excitonic_momentum,excitonic_momentum,0,0,0)),1,0,0);
	cout<<"ending extracting rho"<<endl;
	
	cout<<"fourth part"<<endl;
	cx_mat temporary_matrix2(number_k_points_list,number_g_points_list);
	cx_vec temporary_vector2(number_conduction_bands*number_valence_bands*number_k_points_list);
	cx_double factor_w;
	factor_w.real(-1.0/number_k_points_list);
	factor_w.imag(0.0);
	///double factor_w=-1;
	int spinv1; int spinc1;
	for(int spin1=0;spin1<(3*spinorial_calculation+1);spin1++){
		spinv1=exciton_spin(0,spin1);
		spinc1=exciton_spin(1,spin1);
		///#pragma omp parallel for collapse(3) shared(spin1,spinv1,spinc1) private(temporary_matrix2,temporary_vector2)
		for(int c1=0;c1<number_conduction_bands;c1++)
			for(int v1=0;v1<number_valence_bands;v1++)
				for(int k1=0;k1<number_k_points_list;k1++){
					//cout<<c1<<" "<<v1<<" "<<k1<<endl;
					for(int c2=0;c2<number_conduction_bands;c2++){
						temporary_matrix2=(rho_kk_cc.submat(spinc1*number_conduction_bands*number_conduction_bands*number_k_points_list*number_k_points_list+c1*number_conduction_bands*number_k_points_list*number_k_points_list+c2*number_k_points_list*number_k_points_list+k1*number_k_points_list,0,spinc1*number_conduction_bands*number_conduction_bands*number_k_points_list*number_k_points_list+c1*number_conduction_bands*number_k_points_list*number_k_points_list+c2*number_k_points_list*number_k_points_list+(k1+1)*number_k_points_list-1,number_g_points_list-1)*v_coulomb_gg);
						for(int v2=0;v2<number_valence_bands;v2++)
							temporary_vector2.subvec(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list,c2*number_valence_bands*number_k_points_list+(v2+1)*number_k_points_list-1)=factor_w*diagvec(temporary_matrix2*(rho_qq_kk_vv.submat(spinv1*number_valence_bands*number_valence_bands*number_k_points_list*number_k_points_list+v1*number_valence_bands*number_k_points_list*number_k_points_list+v2*number_k_points_list*number_k_points_list+k1*number_k_points_list,0,spinv1*number_valence_bands*number_valence_bands*number_k_points_list*number_k_points_list+v1*number_valence_bands*number_k_points_list*number_k_points_list+v2*number_k_points_list*number_k_points_list+(k1+1)*number_k_points_list-1,number_g_points_list-1).t()));
					}
					excitonic_hamiltonian.submat(spin1*number_conduction_bands*number_valence_bands*number_k_points_list+c1*number_valence_bands*number_k_points_list+v1*number_k_points_list+k1,
						spin1*number_conduction_bands*number_valence_bands*number_k_points_list,spin1*number_conduction_bands*number_valence_bands*number_k_points_list+c1*number_valence_bands*number_k_points_list+v1*number_k_points_list+k1,
						(spin1+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1)=temporary_vector2.t();
				}
	}
	temporary_matrix2.reset();
	temporary_vector2.reset();
	rho_kk_cc.reset();
	rho_qq_kk_vv.reset();
	
	///separated == 1 gives H00=Resonant Part H11=Coupling Part
	///rewriting energies in order to obtain something that can be summed to the rest
	cout<<"fifth part"<<endl;
	cx_vec energies_q_vc((spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list);
	for(int spin=0;spin<(spinorial_calculation+1);spin++){
		#pragma omp parallel for collapse(2)
		for(int c=0;c<number_conduction_bands;c++)
			for(int v=0;v<number_valence_bands;v++)
				energies_q_vc.subvec(spin*number_conduction_bands*number_valence_bands*number_k_points_list+c*number_valence_bands*number_k_points_list+v*number_k_points_list,spin*number_conduction_bands*number_valence_bands*number_k_points_list+c*number_valence_bands*number_k_points_list+(v+1)*number_k_points_list-1)=energies_q.submat(spin,v*number_conduction_bands*number_k_points_list+c*number_k_points_list,spin,v*number_conduction_bands*number_k_points_list+(c+1)*number_k_points_list-1).t();
	}
	for(int i=0;i<(spinorial_calculation+1);i++)
		for(int r=0;r<number_conduction_bands*number_valence_bands*number_k_points_list;r++)
			excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r)+=-energies_q_vc(i*number_conduction_bands*number_valence_bands*number_k_points_list+r);
	energies_q_vc.reset();

	cout<<"Conjugating HBSE"<<endl;
	//#pragma omp parallel for collapse(2)
	for(int i=0;i<spin_dimension_bse_hamiltonian;i++)
		for(int j=0;j<spin_dimension_bse_hamiltonian;j++)
			excitonic_hamiltonian(spin_dimension_bse_hamiltonian+i,spin_dimension_bse_hamiltonian+j)=-conj(excitonic_hamiltonian(i,j));
	cout<<"building 00 finished"<<endl;
	///cout<<"hermicity h: "<<accu(excitonic_hamiltonian-conj(excitonic_hamiltonian).t())<<endl;
	//cout<<excitonic_hamiltonian<<endl;
};
void Excitonic_Hamiltonian::add_coupling_part(){
	///building entry 01 (coupling part)
	cout<<"building 01"<<endl;
	///building v
	cout<<"building v"<<endl;
	cout<<"beginning extracting rho"<<endl;
	vec zeros_vec(3,fill::zeros);
	cx_mat rho_p_diagk_vc=(dipole_elements->pull_reduced_values_vc(get<1>(dipole_elements->pull_values(excitonic_momentum,-excitonic_momentum,zeros_vec,1,0,0)),1,1)).t();
	cout<<"ending extracting rho"<<endl;
	cx_vec zeros_long_vec((spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list,fill::zeros);
	cx_mat temporary_matrix1(number_g_points_list,(spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list);
	cx_double factor_v;
	factor_v.real((spinorial_calculation-2)/number_k_points_list);
	factor_v.imag(0.0);
	///double factor_v=(spinorial_calculation-2);
	for(int i=0;i<(spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list;i++)
		temporary_matrix1.col(i)=factor_v*(v_coulomb_g%rho_p_diagk_vc.col(i));
	////excluding G=0
	temporary_matrix1.row(0)=zeros_long_vec.t();
	rho_p_diagk_vc.row(0)=zeros_long_vec.t();
	cout<<"second part"<<endl;
	for(int i=0;i<(spinorial_calculation+1);i++)
		for(int j=0;j<(spinorial_calculation+1);j++)
			excitonic_hamiltonian.submat(i*3*number_conduction_bands*number_valence_bands*number_k_points_list,spin_dimension_bse_hamiltonian+j*3*number_conduction_bands*number_valence_bands*number_k_points_list,(i*3+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1,spin_dimension_bse_hamiltonian+(j*3+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1)=
				(temporary_matrix1.submat(0,i*number_conduction_bands*number_valence_bands*number_k_points_list,number_g_points_list-1,(i+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1).t())*(conj(rho_p_diagk_vc.submat(0,j*number_conduction_bands*number_valence_bands*number_k_points_list,number_g_points_list-1,(1+j)*number_conduction_bands*number_valence_bands*number_k_points_list-1)));
	temporary_matrix1.reset();
	rho_p_diagk_vc.reset();
	zeros_long_vec.reset();
	///building W
	cout<<"building w"<<endl;
	cx_double factor_w;
	factor_w.real(-1.0/number_k_points_list);
	factor_w.imag(0.0);
	///double factor_w=-1;
	cout<<"beginning extracting rho"<<endl;
	cx_mat rho_q_kk_vc=dipole_elements->pull_reduced_values_vc(get<1>(dipole_elements->pull_values(excitonic_momentum,excitonic_momentum,zeros_vec,0,0,0)),0,0);
	cx_mat rho_mq_kk_cv=dipole_elements->pull_reduced_values_cv(get<1>(dipole_elements->pull_values(excitonic_momentum,zeros_vec,-excitonic_momentum,0,0,0)),0,0);
	cout<<"ending extracting rho"<<endl;
	cx_mat temporary_matrix3(number_k_points_list,number_g_points_list);
	cx_vec temporary_vector3(number_conduction_bands*number_valence_bands*number_k_points_list);
	int spin2; int spin1;
	int spinv1; int spinc1;
	for(int spin1=0;spin1<(3*spinorial_calculation+1);spin1++){
		spinv1=exciton_spin(0,spin1);
		spinc1=exciton_spin(1,spin1);
		if((spin1==0)||(spin1==3*spinorial_calculation+1))
			spin2=spin1;
		else
			spin2=(2-spin1)+1;
		//#pragma omp parallel for collapse(3) shared(spin1,spinv1,spinc1)
		for(int c1=0;c1<number_conduction_bands;c1++)
			for(int v1=0;v1<number_valence_bands;v1++)
				for(int k1=0;k1<number_k_points_list;k1++){
					for(int c2=0;c2<number_conduction_bands;c2++){
						temporary_matrix3=conj(rho_q_kk_vc.submat(spinv1*number_conduction_bands*number_valence_bands*number_k_points_list*number_k_points_list+v1*number_conduction_bands*number_k_points_list*number_k_points_list+c2*number_k_points_list*number_k_points_list+k1*number_k_points_list,0,spinv1*number_conduction_bands*number_valence_bands*number_k_points_list*number_k_points_list+v1*number_conduction_bands*number_k_points_list*number_k_points_list+c2*number_k_points_list*number_k_points_list+(k1+1)*number_k_points_list-1,number_g_points_list-1))*v_coulomb_gg;
						for(int v2=0;v2<number_valence_bands;v2++)
							temporary_vector3.subvec(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list,c2*number_valence_bands*number_k_points_list+(v2+1)*number_k_points_list-1)=factor_w*diagvec(temporary_matrix3*(rho_mq_kk_cv.submat(spinc1*number_conduction_bands*number_valence_bands*number_k_points_list*number_k_points_list+c1*number_valence_bands*number_k_points_list*number_k_points_list+v2*number_k_points_list*number_k_points_list+k1*number_k_points_list,0,spinc1*number_conduction_bands*number_valence_bands*number_k_points_list*number_k_points_list+c1*number_valence_bands*number_k_points_list*number_k_points_list+v2*number_k_points_list*number_k_points_list+(k1+1)*number_k_points_list-1,number_g_points_list-1).t()));
					}
					excitonic_hamiltonian.submat(spin1*number_conduction_bands*number_valence_bands*number_k_points_list+c1*number_valence_bands*number_k_points_list+v1*number_k_points_list+k1,
						spin_dimension_bse_hamiltonian+spin2*number_conduction_bands*number_valence_bands*number_k_points_list,spin1*number_conduction_bands*number_valence_bands*number_k_points_list+c1*number_valence_bands*number_k_points_list+v1*number_k_points_list+k1,
						spin_dimension_bse_hamiltonian+(spin2+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1)=temporary_vector3.t();
				}
	}
	temporary_matrix3.reset();
	temporary_vector3.reset();
	rho_q_kk_vc.reset();
	rho_mq_kk_cv.reset();

	cout<<"Conjugating HBSE"<<endl;
	//#pragma omp parallel for collapse(2)
	for(int i=0;i<spin_dimension_bse_hamiltonian;i++)
		for(int j=0;j<spin_dimension_bse_hamiltonian;j++)
			excitonic_hamiltonian(spin_dimension_bse_hamiltonian+i,j)=-conj(excitonic_hamiltonian(i,spin_dimension_bse_hamiltonian+j));
	cout<<"building 01 finished"<<endl;
};
tuple<cx_mat,cx_mat> Excitonic_Hamiltonian::extract_hbse_and_rcv(vec excitonic_momentum_tmp,double eta,Coulomb_Potential *coulomb_potential,Dielectric_Function *dielectric_function,int adding_screening,int tamn_dancoff,int order_approximation){
	pull_coulomb_potentials(coulomb_potential,dielectric_function,adding_screening,excitonic_momentum_tmp,eta,order_approximation);
	pull_resonant_part_and_rcv(excitonic_momentum_tmp,eta);
	if(tamn_dancoff==0)
		add_coupling_part();
	return{excitonic_hamiltonian,rho_q_diagk_cv};
};
/// usual diagonalization routine
tuple<cx_vec,cx_mat> Excitonic_Hamiltonian::common_diagonalization(vec excitonic_momentum_tmp,double eta,Coulomb_Potential *coulomb_potential,Dielectric_Function *dielectric_function,int adding_screening,int tamn_dancoff,int order_approximation){
	
	pull_coulomb_potentials(coulomb_potential,dielectric_function,adding_screening,excitonic_momentum_tmp,eta,order_approximation);
	pull_resonant_part_and_rcv(excitonic_momentum_tmp,eta);
	if(tamn_dancoff==0)
		add_coupling_part();
	
	cout<<"diagonalization HBSE"<<endl;
	///diagonalizing the BSE matrix
	///M_{(bz_number_k_points_list x number_valence_bands x number_conduction_bands)x(bz_number_k_points_list x number_valence_bands x number_conduction_bands)}
	if (spinorial_calculation == 1){
		int dimension_bse_hamiltonian_2=spin_dimension_bse_hamiltonian_2/2;
		///SISTEMARE DIMENSIONI
		///SISTEMARE SOTTOMATRICI CONSIDERATE, TENERE CONTO THE COUPLING PARTE E RESONANT PARTE SONO STATE COSTRUITE SEPARATAMENT
		///separating the excitonic hamiltonian in two blocks; still not the ones associated to the magnons and excitons
		cx_mat excitonic_hamiltonian_0(dimension_bse_hamiltonian_2,dimension_bse_hamiltonian_2);
		cx_mat excitonic_hamiltonian_1=excitonic_hamiltonian.submat(number_conduction_bands*number_valence_bands*number_k_points_list,number_conduction_bands*number_valence_bands*number_k_points_list,3*number_conduction_bands*number_valence_bands*number_k_points_list-1,3*number_conduction_bands*number_valence_bands*number_k_points_list-1);
		///cout<<excitonic_hamiltonian_1_tmp.n_cols<<" "<<excitonic_hamiltonian_1_tmp.n_rows<<endl;
		for(int i=0;i<2;i++){
			excitonic_hamiltonian_0.submat(i*dimension_bse_hamiltonian,i*dimension_bse_hamiltonian,(i+1)*dimension_bse_hamiltonian-1,(i+1)*dimension_bse_hamiltonian-1)=
				excitonic_hamiltonian.submat(3*i*number_conduction_bands*number_valence_bands*number_k_points_list,3*i*number_conduction_bands*number_valence_bands*number_k_points_list,(3*i+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1,(3*i+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1);
		}
		///diagonalizing the two spin channels separately: M=0 and M=\pm1
		cx_vec eigenvalues_1(dimension_bse_hamiltonian_2); 
		cx_mat eigenvectors_1(dimension_bse_hamiltonian_2,dimension_bse_hamiltonian_2);
		cx_vec eigenvalues_0(dimension_bse_hamiltonian_2); 
		cx_mat eigenvectors_0(dimension_bse_hamiltonian_2,dimension_bse_hamiltonian_2);
		lapack_complex_double *temporary_0;
		temporary_0=(lapack_complex_double*)malloc(dimension_bse_hamiltonian_2*dimension_bse_hamiltonian_2*sizeof(lapack_complex_double)); 
		lapack_complex_double *temporary_1;
		temporary_1=(lapack_complex_double*)malloc(dimension_bse_hamiltonian_2*dimension_bse_hamiltonian_2*sizeof(lapack_complex_double)); 

		#pragma omp parallel for collapse(2)
		for(int i=0;i<dimension_bse_hamiltonian_2;i++)
			for(int j=0;j<dimension_bse_hamiltonian_2;j++){
				temporary_0[i*dimension_bse_hamiltonian_2+j]=real(excitonic_hamiltonian_0(i,j))+_Complex_I*imag(excitonic_hamiltonian_0(i,j));
				temporary_1[i*dimension_bse_hamiltonian_2+j]=real(excitonic_hamiltonian_1(i,j))+_Complex_I*imag(excitonic_hamiltonian_1(i,j));
			}
	
		int N=dimension_bse_hamiltonian_2;
		int LDA=dimension_bse_hamiltonian_2;
		int LDVL=1;
		int LDVR=dimension_bse_hamiltonian_2;
		char JOBVR='V';
		char JOBVL='N';
		int matrix_layout = 101;
		int INFO0; int INFO1;
		lapack_complex_double *empty;

		lapack_complex_double *w_0;
		w_0=(lapack_complex_double*)malloc(N*sizeof(lapack_complex_double));
		lapack_complex_double *u_0;
		u_0=(lapack_complex_double*)malloc(N*LDVR*sizeof(lapack_complex_double));
		lapack_complex_double *w_1;
		w_1=(lapack_complex_double*)malloc(N*sizeof(lapack_complex_double));
		lapack_complex_double *u_1;
		u_1=(lapack_complex_double*)malloc(N*LDVR*sizeof(lapack_complex_double));
		
		INFO0 = LAPACKE_zgeev(matrix_layout,JOBVL,JOBVR,N,temporary_0,LDA,w_0,empty,LDVL,u_0,LDVR);
		INFO1 = LAPACKE_zgeev(matrix_layout,JOBVL,JOBVR,N,temporary_1,LDA,w_1,empty,LDVL,u_1,LDVR);
		
		free(temporary_0); free(temporary_1);
		//eig_gen(eigenvalues_1,eigenvectors_1,excitonic_hamiltonian_1);
		//eig_gen(eigenvalues_0,eigenvectors_0,excitonic_hamiltonian_0);
		for(int i=0;i<dimension_bse_hamiltonian_2;i++){
			eigenvalues_0(i).real(lapack_complex_double_real(w_0[i]));
			eigenvalues_0(i).imag(lapack_complex_double_imag(w_0[i]));
			eigenvalues_1(i).real(lapack_complex_double_real(w_1[i]));
			eigenvalues_1(i).imag(lapack_complex_double_imag(w_1[i]));
		}

		///ordering the eigenvalues and saving them in a single matrix exc_eigenvalues
		cx_vec exc_eigenvalues(spin_dimension_bse_hamiltonian); uvec ordering_0=sort_index(eigenvalues_0);
		cx_mat exc_eigenvectors(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian);
		/// normalizing and ordering eigenvectors: saving them in a single matrix exc_eigenvectors
		for(int i=0;i<dimension_bse_hamiltonian_2;i++){
			for(int s=0;s<dimension_bse_hamiltonian_2;s++)
				exc_eigenvectors(s,i)=u_0[s*dimension_bse_hamiltonian_2+ordering_0(i)]; 
			exc_eigenvalues(i) = eigenvalues_0(ordering_0(i));
		}

		/// separating magnons and excitons
		///to add routine separating the two parts
		uvec ordering_1=sort_index(eigenvalues_1.subvec(0,dimension_bse_hamiltonian));
		for(int i=0;i<dimension_bse_hamiltonian;i++){
			for(int s=0;s<dimension_bse_hamiltonian;s++)
				exc_eigenvectors(s+dimension_bse_hamiltonian_2,i+dimension_bse_hamiltonian_2)=u_1[s*dimension_bse_hamiltonian+ordering_1(i)]; 
			exc_eigenvalues(i+dimension_bse_hamiltonian_2) = eigenvalues_1(ordering_1(i));
		}
		for(int i=0;i<dimension_bse_hamiltonian;i++){
			for(int s=0;s<dimension_bse_hamiltonian;s++)
			exc_eigenvectors(s+dimension_bse_hamiltonian_2+dimension_bse_hamiltonian,i+dimension_bse_hamiltonian_2+dimension_bse_hamiltonian)=u_1[(s+1)*dimension_bse_hamiltonian+i+dimension_bse_hamiltonian]; 
			exc_eigenvalues(i+dimension_bse_hamiltonian_2+dimension_bse_hamiltonian) = eigenvalues_1(i+dimension_bse_hamiltonian);
		}
		for(int i=0;i<spin_dimension_bse_hamiltonian;i++)
			exc_eigenvectors.col(i)=exc_eigenvectors.col(i)/norm(exc_eigenvectors.col(i),2);

		free(w_0); free(w_1); free(u_0); free(u_1);
		return {exc_eigenvalues,exc_eigenvectors};
	}else{
		///lapack_complex_double *temporary;
		///temporary=(lapack_complex_double*)malloc(spin_dimension_bse_hamiltonian_2*spin_dimension_bse_hamiltonian_2*sizeof(lapack_complex_double));
		///for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++)
		///	for(int j=0;j<spin_dimension_bse_hamiltonian_2;j++){
		///		temporary[i*spin_dimension_bse_hamiltonian_2+j]=real(excitonic_hamiltonian(i,j))+_Complex_I*imag(excitonic_hamiltonian(i,j));	
		///}
		///	
		///int N=spin_dimension_bse_hamiltonian_2;
		///int LDA=spin_dimension_bse_hamiltonian_2;
		///int LDVL=spin_dimension_bse_hamiltonian_2;
		///int LDVR=spin_dimension_bse_hamiltonian_2;
		///char JOBVR='N';
		///char JOBVL='V';
		///int matrix_layout = 101;
		///int INFO;
		///lapack_complex_double *w;
		///w=(lapack_complex_double*)malloc(N*sizeof(lapack_complex_double));
		///lapack_complex_double *u;
		///u=(lapack_complex_double*)malloc(N*LDVR*sizeof(lapack_complex_double));
		///lapack_complex_double *empty;
		///empty=(lapack_complex_double*)malloc(N*LDVL*sizeof(lapack_complex_double));
		///
		///INFO = LAPACKE_zgeev(matrix_layout,JOBVL,JOBVR,N,temporary,LDA,w,u,LDVR,empty,LDVL);
		///
		///free(temporary);
		///free(empty); 
		///cx_vec eigenvalues(spin_dimension_bse_hamiltonian_2);
		///for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++){
		///	eigenvalues(i).real(lapack_complex_double_real(w[i]));
		///	eigenvalues(i).imag(lapack_complex_double_imag(w[i]));
		///}
		//for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++){
		//	for(int s=0;s<spin_dimension_bse_hamiltonian_2;s++){
		//		exc_eigenvectors(ordering(i),s).real(lapack_complex_double_real(u[s*spin_dimension_bse_hamiltonian_2+ordering(i)]));
		//		exc_eigenvectors(ordering(i),s).imag(lapack_complex_double_imag(u[s*spin_dimension_bse_hamiltonian_2+ordering(i)]));
		//	}
		//	exc_eigenvalues(ordering(i))=eigenvalues(i);
		//}
		cx_vec eigenvalues; cx_mat eigenvectors;
		eig_gen(eigenvalues,eigenvectors,excitonic_hamiltonian);
		//cout<<excitonic_hamiltonian<<endl;
		cx_vec exc_eigenvalues(spin_dimension_bse_hamiltonian_2); 
		cx_mat exc_eigenvectors(spin_dimension_bse_hamiltonian_2,spin_dimension_bse_hamiltonian_2);
		cx_double exc_norm;
		uvec ordering=sort_index(real(eigenvalues));
		for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++){
			exc_norm=norm(eigenvectors.col(i),2);
			//cout<<eigenvectors.col(i)<<" ";
			for(int s=0;s<spin_dimension_bse_hamiltonian_2;s++)
				if(real(exc_norm)!=0.0||imag(exc_norm)!=0.0)
					exc_eigenvectors(s,ordering(i))=eigenvectors(s,i)/exc_norm;
				else
					exc_eigenvectors(s,ordering(i))=eigenvectors(s,i);
			exc_eigenvalues(ordering(i))=eigenvalues(i);
			///cout<<exc_eigenvectors.col(ordering(i))<<endl;
		}
		//cout<<exc_eigenvalues<<" ";
		//for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++)
		//	cout<<i<<" "<<exc_eigenvectors.col(i)<<" ";
		return {exc_eigenvalues.subvec(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian_2-1),exc_eigenvectors.submat(0,spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian_2-1)};
	}
};

/// Fastest diagonalization routine
///[1] Structure preserving parallel algorithms for solving the Bethe–Salpeter eigenvalue problem Meiyue Shao, Felipe H. da Jornada, Chao Yang, Jack Deslippe, Steven G. Louie
///[2] Beyond the Tamm-Dancoff approximation for ext.ended systems using exact diagonalization Tobias Sander, Emanuelel Maggio, and Georg Kresse
tuple<vec,cx_mat> Excitonic_Hamiltonian:: cholesky_diagonalization(vec excitonic_momentum_tmp,double eta,Coulomb_Potential *coulomb_potential,Dielectric_Function *dielectric_function,int adding_screening,int order_approximation)
{
	pull_coulomb_potentials(coulomb_potential,dielectric_function,adding_screening,excitonic_momentum_tmp,eta,order_approximation);
	pull_resonant_part_and_rcv(excitonic_momentum_tmp,eta);
	add_coupling_part();

	/// diagonalizing the BSE matrix M_{(bz_number_k_points_list x number_valence_bands x number_conduction_bands)x(bz_number_k_points_list x number_valence_bands x number_conduction_bands)}
	int spin_dimension_bse_hamiltonian_2 = 2*spin_dimension_bse_hamiltonian;
	cx_mat A=excitonic_hamiltonian.submat(0,0,spin_dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian-1);
	cx_mat B=excitonic_hamiltonian.submat(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian,2*spin_dimension_bse_hamiltonian-1,2*spin_dimension_bse_hamiltonian-1);

	mat M(spin_dimension_bse_hamiltonian_2, spin_dimension_bse_hamiltonian_2);
	// Fill top-left submatrix
	M.submat(0, 0, spin_dimension_bse_hamiltonian - 1, spin_dimension_bse_hamiltonian - 1) = real(A + B);
	// Fill bottom-left submatrix (as conjugate transpose of top-right submatrix)
	M.submat(spin_dimension_bse_hamiltonian, 0, spin_dimension_bse_hamiltonian_2 - 1, spin_dimension_bse_hamiltonian - 1) = -imag(A + B).t();
	// Fill top-right submatrix (as conjugate transpose of bottom-left submatrix)
	M.submat(0, spin_dimension_bse_hamiltonian, spin_dimension_bse_hamiltonian - 1, spin_dimension_bse_hamiltonian_2 - 1) = -imag(A-B);
	// Fill bottom-right submatrix
	M.submat(spin_dimension_bse_hamiltonian, spin_dimension_bse_hamiltonian, spin_dimension_bse_hamiltonian_2 - 1, spin_dimension_bse_hamiltonian_2 - 1) = real(A - B);
	
	///symmetrizing
	M+=M.t();
	M=M/2;
	cout<<M.is_symmetric()<<endl;

	cout<<"cholesky factorization"<<endl;
	/// construct W
	vec diag_one(spin_dimension_bse_hamiltonian,fill::ones);
	mat J(spin_dimension_bse_hamiltonian_2,spin_dimension_bse_hamiltonian_2,fill::zeros);
	J.submat(0,spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian_2-1)=diagmat(diag_one);
	J.submat(spin_dimension_bse_hamiltonian,0,spin_dimension_bse_hamiltonian_2-1,spin_dimension_bse_hamiltonian-1)=-diagmat(diag_one);
	
	mat L(spin_dimension_bse_hamiltonian_2, spin_dimension_bse_hamiltonian_2,fill::zeros);
	//L=chol(M);
	
	int N=spin_dimension_bse_hamiltonian_2;
	int LDA=spin_dimension_bse_hamiltonian_2;
	int matrix_layout = 101;
	int INFO;
	char UPLO = 'U';
	double *temporary_0; temporary_0 = (double *)malloc(spin_dimension_bse_hamiltonian_2*spin_dimension_bse_hamiltonian_2*sizeof(double));
	#pragma omp parallel for collapse(2)
	for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++)
		for(int j=0;j<spin_dimension_bse_hamiltonian_2;j++){
			temporary_0[i*spin_dimension_bse_hamiltonian+j]=M(i,j);
		}

	INFO=LAPACKE_dpotrf(matrix_layout, UPLO, N, temporary_0, LDA);

	for (int i = 0; i < spin_dimension_bse_hamiltonian_2; i++)
		for (int j = i; j < spin_dimension_bse_hamiltonian_2; j++){
			L(i,j)=temporary_0[i * spin_dimension_bse_hamiltonian_2 + j];
		}

	free(temporary_0);

	mat W(spin_dimension_bse_hamiltonian_2,spin_dimension_bse_hamiltonian_2);
	W =  L.t() * J;
	W = W * L;
	
	lapack_complex_float *temporary_1; temporary_1 = (lapack_complex_float *)malloc(spin_dimension_bse_hamiltonian_2*spin_dimension_bse_hamiltonian_2*sizeof(lapack_complex_float));
	
	#pragma omp parallel for collapse(2)
	for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++)
		for(int j=0;j<spin_dimension_bse_hamiltonian_2;j++){
			temporary_1[i*spin_dimension_bse_hamiltonian_2+j]=imag(W(i,j))-_Complex_I*real(W(i,j));
			//cout<<temporary_1[i*spin_dimension_bse_hamiltonian_2+j]<<endl;
			cout<<W(i,j)<<" ";
		}

	float *w;
	char JOBZ = 'V';
	char JOBU = 'A';
	char JOBVT = 'A';
	int F=spin_dimension_bse_hamiltonian_2;
	int LDU=spin_dimension_bse_hamiltonian_2;
	int LDVT=spin_dimension_bse_hamiltonian_2;
	lapack_complex_float *z1;
	lapack_complex_float *z2;
	z1=(lapack_complex_float*)malloc(F*F*sizeof(lapack_complex_float));
	z2=(lapack_complex_float*)malloc(F*F*sizeof(lapack_complex_float));
	//// saving all the eigenvalues
	w = (float *)malloc(spin_dimension_bse_hamiltonian_2 * sizeof(float));
	float *lwork; lwork=(float*)malloc(4*F*sizeof(float));

	cout<<"singular value decomposition"<<endl;
	INFO=LAPACKE_cgesvd(matrix_layout,JOBU,JOBVT,F,N,temporary_1,LDA,w,z1,LDU,z2,LDVT,lwork);
	
	free(temporary_1);

	vec exc_eigenvalues(spin_dimension_bse_hamiltonian);
	for(int i=0;i<spin_dimension_bse_hamiltonian;i++){
		exc_eigenvalues(i)=-w[i];
		cout<<exc_eigenvalues(i)<<endl;
	}
	free(w);

	cx_mat exc_eigenvectors(spin_dimension_bse_hamiltonian_2,spin_dimension_bse_hamiltonian_2);
	cx_mat z_(spin_dimension_bse_hamiltonian_2,spin_dimension_bse_hamiltonian_2);
	for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++)
		for(int j=0;j<spin_dimension_bse_hamiltonian_2;j++)
			z_(i,j)=z1[i*spin_dimension_bse_hamiltonian_2+j];
	free(z1); free(z2);
		
	cx_double ieta;
	ieta.real(0.0); ieta.imag(eta);

	exc_eigenvectors.set_real(J*L);
	exc_eigenvectors=exc_eigenvectors*z_;

	exc_eigenvectors.submat(0,0,spin_dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian-1)=exc_eigenvectors.submat(0,0,spin_dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian-1)*inv(diagmat(sqrt(exc_eigenvalues)+ieta));
	exc_eigenvectors.submat(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian_2-1,spin_dimension_bse_hamiltonian_2-1)=exc_eigenvectors.submat(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian_2-1,spin_dimension_bse_hamiltonian_2-1)*inv(diagmat(sqrt(exc_eigenvalues)+ieta));

	return{exc_eigenvalues,exc_eigenvectors.submat(0,0,spin_dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian-1)};
};

cx_vec Excitonic_Hamiltonian::pull_excitonic_oscillator_force(cx_mat excitonic_eigenstates){
	cx_vec oscillator_force(dimension_bse_hamiltonian,fill::zeros);
	int g_point_0=int(number_g_points_list/2);

	if(spinorial_calculation==1){
		cx_mat excitonic_eigenstates_reduced=excitonic_eigenstates.submat(0,0,2*dimension_bse_hamiltonian-1,2*dimension_bse_hamiltonian-1);	
		for(int i=0;i<dimension_bse_hamiltonian;i++){
			//excitonic_eigenstates_reduced.col(i)=excitonic_eigenstates_reduced.col(i)%multiplicative_factor;
			oscillator_force(i)=accu(rho_q_diagk_cv.col(g_point_0)%conj(excitonic_eigenstates_reduced.col(i)));
		}
	}else
		for(int i=0;i<dimension_bse_hamiltonian;i++){
			//cout<<"eigenstates  "<<excitonic_eigenstates.col(i)<<endl;
			oscillator_force(i)=accu(conj(rho_q_diagk_cv.col(g_point_0))%excitonic_eigenstates.col(i));
		}

	return oscillator_force;
};
void Excitonic_Hamiltonian:: pull_macroscopic_bse_dielectric_function(cx_vec omegas_path,int number_omegas_path,double eta,string file_macroscopic_dielectric_function_bse_name,double lorentzian,int tamn_dancoff,Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,int order_approximation)
{
	//cout << "Calculating dielectric tensor..." << endl;
	double factor=8*pigreco/(minval*minval);

	cx_cube dielectric_tensor_bse(3,3,number_omegas_path,fill::zeros);
	cx_double ieta;	ieta.real(0.0);	ieta.imag(eta);
	cx_double ilorentzian; ilorentzian.real(0.0); ilorentzian.imag(lorentzian);
	cx_vec temporary_variable(number_omegas_path);
	vec excitonic_momentum1(3,fill::zeros);

	cx_mat delta(3,3,fill::zeros);
	for(int i=0;i<3;i++)
		delta(i,i).real(1.0);

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			if(i==j&&i==0){
				excitonic_momentum1(i)+=minval;
				tuple<cx_vec,cx_mat> eigenvalues_and_eigenstates = common_diagonalization(excitonic_momentum1,eta,coulomb_potential,dielectric_function,1,tamn_dancoff,order_approximation);
				cx_vec exc_eigenvalues=get<0>(eigenvalues_and_eigenstates);
				cx_mat exc_eigenstates=get<1>(eigenvalues_and_eigenstates);
				///cout<<exc_eigenstates<<" ";
				cx_vec exc_oscillator_force1=pull_excitonic_oscillator_force(exc_eigenstates);
				excitonic_momentum1(i)-=minval;
				for(int s=0;s<number_omegas_path;s++){
					temporary_variable(s)=0.0;
					for(int l=0;l<dimension_bse_hamiltonian;l++)
						temporary_variable(s)+=exc_oscillator_force1(l)*conj(exc_oscillator_force1(l))/(omegas_path(s)-exc_eigenvalues(l)+ilorentzian);
					temporary_variable(s).imag(factor*temporary_variable(s).imag());
					dielectric_tensor_bse(i,j,s)=delta(i,j)-temporary_variable(s);
					cout<<dielectric_tensor_bse(i,j,s)<<endl;
				}
				exc_eigenvalues.reset();
				exc_eigenstates.reset();
				(get<0>(eigenvalues_and_eigenstates)).reset();
				(get<1>(eigenvalues_and_eigenstates)).reset();
				exc_oscillator_force1.reset();
			}

	ofstream dielectric_tensor_file;
	dielectric_tensor_file.open(file_macroscopic_dielectric_function_bse_name);
	///writing the dielectric function (in the optical limit) in a file
	dielectric_tensor_file<<"### omega xx xy xz yx yy yz zx zy zz"<<endl;
	for(int s=0;s<number_omegas_path;s++){
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				dielectric_tensor_file<<omegas_path(s)<<" "<<dielectric_tensor_bse(i,j,s)<<" ";
		dielectric_tensor_file<<endl;
	}
	dielectric_tensor_file.close();
};
void Excitonic_Hamiltonian::print(vec excitonic_momentum_tmp,double eta,int tamn_dancoff,Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,int order_approximation){
	
	pull_coulomb_potentials(coulomb_potential,dielectric_function,adding_screening,excitonic_momentum,eta,order_approximation);	
	pull_resonant_part_and_rcv(excitonic_momentum_tmp,eta);
	if(tamn_dancoff==0)
		add_coupling_part();
	cout<<number_k_points_list<<endl;
	
	cout<<"BSE hamiltoian..."<<endl;
	for(int i=0;i<2*spin_dimension_bse_hamiltonian;i++){
		for(int j=0;j<2*spin_dimension_bse_hamiltonian;j++)
			cout<<excitonic_hamiltonian(i,j)<<" ";
		cout<<endl;
	}
	
	///cout<<"Eigenvalues..."<<endl;
	///cx_vec eigenvalues(spin_dimension_bse_hamiltonian);
	///cx_mat eigenstates(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian);
	///cx_mat rho((spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list,number_g_points_list);
	///tuple<cx_vec,cx_mat,cx_mat> eigenvalues_and_eigenstates_and_rho_cv(eigenvalues,eigenstates,rho);
	///eigenvalues_and_eigenstates_and_rho_cv=common_diagonalization(excitonic_momentum,eta,coulomb_potential,dielectric_function,adding_screening,tamn_dancoff);
	///eigenvalues=get<0>(eigenvalues_and_eigenstates_and_rho_cv);
	///eigenstates=get<1>(eigenvalues_and_eigenstates_and_rho_cv);
	///for (int i=0;i<spin_dimension_bse_hamiltonian;i++)
	///	cout<<eigenvalues(i)<<endl;
};

///void diagonalize_parallel(int n, scalapackpp::Grid& grid, std::vector<std::complex<double>>& local_A, int local_rows, int local_cols) {
///    std::vector<double> eigenvalues(n);
///    std::vector<std::complex<double>> work(1);
///    std::vector<int> iwork(1);
///
///    // Query for workspace size
///    int lwork = -1, liwork = -1;
///    int info = scalapackpp::wrapper::pzheevd(
///        'V', 'U', n, local_A.data(), 1, 1, grid.desc().data(),
///        eigenvalues.data(), work.data(), lwork, iwork.data(), liwork
///    );
///
///    lwork = static_cast<int>( std::real(work[0]) );
///    liwork = iwork[0];
///
///    work.resize(lwork);
///    iwork.resize(liwork);
///
///    // Eigenvalue and eigenvector computation
///    info = scalapackpp::wrapper::pzheevd(
///        'V', 'U', n, local_A.data(), 1, 1, grid.desc().data(),
///        eigenvalues.data(), work.data(), lwork, iwork.data(), liwork
///    );
///
///    if (info == 0) {
///        int my_rank;
///        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
///        if (my_rank == 0) {
///            std::cout << "Parallel eigenvalues:\n";
///            for (const auto& val : eigenvalues) {
///                std::cout << val << "\n";
///            }
///        }
///        // To print eigenvectors, gather them to the root process and print
///        // This part is more complex and might need specific handling depending on the grid configuration and data layout
///    } else {
///        std::cerr << "pzheevd failed with info = " << info << "\n";
///    }
///}

int main(int argc, char** argv)
{
	//MPI_Init(&argc, &argv);
    //int my_rank, num_procs;
    //MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    //MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	//int n;
	//int n_2;

	//if(my_rank==0){
	double fermi_energy = 5.5427;
	////Initializing Lattice
	string file_crystal_bravais_name="bravais.lattice_si.data";
	string file_crystal_coordinates_name="atoms_si.data";
	int number_atoms=2;
	Crystal_Lattice crystal(file_crystal_bravais_name,file_crystal_coordinates_name,number_atoms);
	double volume=crystal.pull_volume();
	mat bravais_lattice=crystal.pull_bravais_lattice();
	crystal.print();

	////Initializing k points list
	vec shift; shift.zeros(3);
	////shift in crystal coordinates
	///shift(0)=0.000001;
	///shift(1)=0.000001;
	///shift(1)=0.000001;
	shift(0)=0.001;
	shift(1)=0.001;
	shift(1)=0.001;
	K_points k_points(&crystal,shift);
	string file_k_points_name="k_points_list_si.6x6x6.dat";
	int number_k_points_list=216;
	int crystal_coordinates=1;
	k_points.push_k_points_list_values(file_k_points_name,number_k_points_list,crystal_coordinates);
	mat k_points_list=k_points.pull_k_points_list_values();
	k_points.print();

	//////Initializing g points list
	vec shift_g; shift_g.zeros(3);
	//double cutoff_g_points_list=2;
	double cutoff_g_points_list=2;
	int dimension_g_points_list=3;
	vec direction_cutting(3); direction_cutting(0)=1; direction_cutting(1)=1; direction_cutting(2)=1;
	G_points g_points(&crystal,cutoff_g_points_list,dimension_g_points_list,direction_cutting,shift_g);
	mat g_points_list=g_points.pull_g_points_list_values(); 
	int number_g_points_list=g_points.pull_number_g_points_list();
	cout<<"G points: "<<number_g_points_list<<endl;
	int g_point0=int(number_g_points_list/2);
	cout<<g_points_list.col(g_point0)<<endl;
	///g_points.print();

	//////Initializing Coulomb potential
	double minimum_k_point_modulus=0.0;
	int dimension_potential=3;
	double radius=0.0;
	Coulomb_Potential coulomb_potential(&k_points,&g_points,minimum_k_point_modulus,dimension_potential,direction_cutting,volume,radius);
	//string file_coulomb_name="coulomb.dat"; 
	//int number_k_points_c=10000; double max_k_points_radius_c=6.0;
	//int direction_profile_xyz=0;
	//coulomb_potential.print_profile(number_k_points_c,max_k_points_radius_c,file_coulomb_name,direction_profile_xyz);

	////Initializing the Tight Binding hamiltonian (saving the Wannier functions centers)
	ifstream file_htb; ifstream file_centers; string seedname;
	string wannier90_hr_file_name="tb_no_spin_polarized.dat";
	string wannier90_centers_file_name="tb_no_spin_polarized_centers.dat";
	bool dynamic_shifting=false;
	int spinorial_calculation = 0;
	double little_shift=0.00;
	double scissor_operator=0.00;
	Hamiltonian_TB htb(wannier90_hr_file_name,wannier90_centers_file_name,fermi_energy,spinorial_calculation,number_atoms,dynamic_shifting,little_shift,scissor_operator,bravais_lattice);
	/// 0 no spinors, 1 collinear spinors, 2 non-collinear spinors (implementing 0 and 1 cases)
	int number_wannier_centers=htb.pull_number_wannier_functions();
	int htb_basis_dimension=htb.pull_htb_basis_dimension();
	//cout<<number_wannier_centers<<endl;
	//htb.print();
	//vec k_point; k_point.zeros(3); k_point(0)=1.0;
	//htb.FFT(k_point);
	//htb.print_ks_states(k_point,4,4);

	//////Initializing dipole elements
	int number_conduction_bands_selected=4;
	int number_valence_bands_selected=4;

	Dipole_Elements dipole_elements(number_k_points_list,k_points_list,number_g_points_list,g_points_list,number_wannier_centers,number_valence_bands_selected,number_conduction_bands_selected,&htb,spinorial_calculation);
	///dipole_elements.print(excitonic_momentum,excitonic_momentum,3,0,0);
	/////////Initializing dielectric function
	Dielectric_Function dielectric_function(&dipole_elements,number_k_points_list,number_g_points_list,g_points_list,number_valence_bands_selected,number_conduction_bands_selected,&coulomb_potential,spinorial_calculation);
	cx_double omega; omega.real(0.0); omega.imag(0.0); double eta=10e-3; 
	////double PPA=27.00;
	int order_approximation=1;
	int number_omegas_path=600;
	cx_vec omegas_path(number_omegas_path);
	cx_double max_omega=8.00;
	cx_double min_omega=0.00;
	cx_vec macroscopic_dielectric_function(number_omegas_path);
	for(int i=0;i<number_omegas_path;i++)
		omegas_path(i)=min_omega+double(i)/double(number_omegas_path)*(max_omega-min_omega);
	///string file_macroscopic_dielectric_function_name="macroscopic_diel_func.dat";
	///dielectric_function.pull_macroscopic_value(omegas_path,number_omegas_path,eta,file_macroscopic_dielectric_function_name,order_approximation);

	////////Initializing BSE hamiltonian
	int adding_screening=1;
	double lorentzian=1.0e-1;
	int tamn_dancoff=1;
	Excitonic_Hamiltonian htbse(number_valence_bands_selected,number_conduction_bands_selected,k_points_list,number_k_points_list,g_points_list,number_g_points_list,spinorial_calculation,htb_basis_dimension,&dipole_elements,volume);
	vec excitonic_momentum; excitonic_momentum.zeros(3);
	excitonic_momentum(0)=minval;
	//htbse.print(excitonic_momentum,eta,tamn_dancoff,&coulomb_potential,&dielectric_function,adding_screening,order_approximation);
	//////calculation optical spectrum
	//vec excitonic_momentum; excitonic_momentum.zeros(3); excitonic_momentum(0)=minval;
	//tuple<cx_mat,cx_mat> excitonic_hamiltonian_and_rho=htbse.extract_hbse_and_rcv(excitonic_momentum_tmp,eta,&coulomb_potential,&dielectric_function,adding_screening,tamn_dancoff,order_approximation);
	//cx_mat excitonic_hamiltonian=get<1>(excitonic_hamiltonian_and_rho);
	//cx_mat rho=get<0>(excitonic_hamiltonian_and_rho);
	
	//n=(spinorial_calculation*3+1)*number_conduction_bands_selected*number_valence_bands_selected*number_k_points_list;
	//n_2=n*2;
	//}
	//int block_size =  int(n_2/2);
	//MPI_Barrier(MPI_COMM_WORLD);
	////Parallel computation using ScaLAPACK
    //scalapackpp::Grid grid( MPI_COMM_WORLD, int(n_2/2), int(n_2/2) );
    //Determine local matrix sizes
    //auto local_dims = scalapackpp::get_local_dims( n_2, n_2, grid, block_size, block_size );
    //int local_rows = local_dims.first;
    //int local_cols = local_dims.second;
	//std::vector<std::complex<double>> local_A(n_2 * n_2);
	//for(int i=0;i<n_2;i++)
	//	for(int j=0;j<n_2;j++)
	//		local_A(i,j)=excitonic_hamiltonian(i,j);
    //diagonalize_parallel(n_2, grid, local_A, local_rows, local_cols);
    //MPI_Finalize();

	string file_macroscopic_dielectric_function_bse_name="macroscopic_diel_func_bse.6x6x6.dat_final";
	htbse.pull_macroscopic_bse_dielectric_function(omegas_path,number_omegas_path,eta,file_macroscopic_dielectric_function_bse_name,lorentzian,tamn_dancoff,&coulomb_potential,&dielectric_function,1,order_approximation);
	
	return 1;
}