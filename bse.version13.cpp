#include <iostream>
#include <string>
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
#include <iomanip>
#include <random>

using namespace std;

// declaring C function/libraries in the C++ code
extern "C"
{
#include <stdio.h>
#include <omp.h>
#include <complex.h>
}

///CONSTANT
const double minval = 1.0e-6;
const double pigreco = 3.1415926535897932384626433832;
///being bravais lattice in Ang, k points are in crystal coordinates and then transformed in Ang^{-1}
const double conversion_parameter = (1.602176634*1000)/(8.8541878176);

////FUNCTIONS
arma::vec cross_product_3(arma::vec a,arma::vec b){
	arma::vec c(3);
	c(0)=a(1)*b(2)-a(2)*b(1);
	c(1)=a(2)*b(0)-a(0)*b(2);
	c(2)=a(0)*b(1)-a(1)*b(0);
	return c;
};

/// START DEFINITION DIFFERENT CLASSES
/// Crystal_Lattice class
class Crystal_Lattice
{
private:
	int number_atoms;
	double volume;
	arma::mat atoms_coordinates;
	arma::mat bravais_lattice{arma::mat(3,3)};
	arma::mat primitive_vectors{arma::mat(3,3)};
public:
	Crystal_Lattice(string bravais_lattice_file_name,string atoms_coordinates_file_name,int number_atoms_tmp);
	arma::vec pull_sitei_coordinates(int sitei);
	arma::mat pull_bravais_lattice();
	arma::mat pull_primitive_vectors();
	arma::mat pull_atoms_coordinates();
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
arma::vec Crystal_Lattice::pull_sitei_coordinates(int sitei){
	return atoms_coordinates.col(sitei);
};
arma::mat Crystal_Lattice::pull_bravais_lattice(){
	return bravais_lattice;
};
arma::mat Crystal_Lattice::pull_atoms_coordinates(){
	return atoms_coordinates;
};
double Crystal_Lattice::pull_volume(){
	return volume;
};
arma::mat Crystal_Lattice::pull_primitive_vectors(){
	return primitive_vectors;
};
Crystal_Lattice::Crystal_Lattice(std::string bravais_lattice_file_name,std::string atoms_coordinates_file_name,int number_atoms_tmp):
atoms_coordinates(3,number_atoms_tmp)
{
	ifstream bravais_lattice_file;
	bravais_lattice_file.open(bravais_lattice_file_name);
	ifstream atoms_coordinates_file;
	atoms_coordinates_file.open(atoms_coordinates_file_name);
	
	bravais_lattice_file.seekg(0);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			bravais_lattice_file >> bravais_lattice(j, i);
	
	number_atoms=number_atoms_tmp;
	atoms_coordinates_file.seekg(0);
	for (int i=0;i<number_atoms;i++)
		for (int j=0;j<3;j++)
			atoms_coordinates_file>>atoms_coordinates(j, i);
			
	volume = std::abs(arma::det(bravais_lattice));

	arma::vec b0(3);
	arma::vec b1(3);
	arma::vec b2(3);
	for(int i=0;i<3;i++){
		b0(i)=bravais_lattice(i,0);
		b1(i)=bravais_lattice(i,1);
		b2(i)=bravais_lattice(i,2);
	}
	primitive_vectors.col(0) = arma::cross(b1,b2);
	primitive_vectors.col(1) = arma::cross(b2,b0);
	primitive_vectors.col(2) = arma::cross(b0,b1);
	double factor = 2 * pigreco / volume;

	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			primitive_vectors(i, j) = factor*primitive_vectors(i, j);
	bravais_lattice_file.close();
	atoms_coordinates_file.close();
};
void Crystal_Lattice::print()
{
	std::cout << "Bravais Lattice:" << endl;
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
	arma::mat primitive_vectors{arma::mat(3,3)};
	arma::mat k_points_list;
	arma::vec shift{arma::vec(3)}; 
	arma::vec direction_cutting{arma::vec(3)};
	arma::mat k_point_differences;
public:
	K_points(Crystal_Lattice *crystal_lattice,arma::vec shift_tmp,int number_k_points_list_tmp);
	void push_k_points_list_values(string k_points_list_file_name,int crystal_coordinates,int random_generator);
	///void push_k_points_list_values(double spacing_tmp,int dimension_tmp,arma::vec direction_cutting_tmp);
	int pull_number_k_points_list();
	arma::mat pull_k_points_list_values();
	arma::mat pull_primitive_vectors();
	arma::vec pull_shift();
	arma::mat pull_k_point_differences(){
		return k_point_differences;
	}
	void print();
	~K_points(){
		spacing=0;
		number_k_points_list=0;
	};
};
K_points::K_points(Crystal_Lattice *crystal_lattice,arma::vec shift_tmp,int number_k_points_list_tmp):
k_points_list(3,number_k_points_list_tmp), k_point_differences(3,number_k_points_list_tmp*number_k_points_list_tmp)
{
	number_k_points_list=number_k_points_list_tmp;
	shift=shift_tmp;
	primitive_vectors=crystal_lattice->pull_primitive_vectors();
};
arma::mat K_points::pull_primitive_vectors(){
	return primitive_vectors;
};
void K_points::push_k_points_list_values(string k_points_list_file_name,int crystal_coordinates,int random_generator){
	cout<<"number points "<<number_k_points_list<<endl;
	if(random_generator==0){
		ifstream k_points_list_file;
		k_points_list_file.open(k_points_list_file_name);
		k_points_list_file.seekg(0);
		int counting = 0;
		while (k_points_list_file.peek()!=EOF){
			if (counting<number_k_points_list)
			{
				k_points_list_file >> k_points_list(0, counting);
				k_points_list_file >> k_points_list(1, counting);
				k_points_list_file >> k_points_list(2, counting);
				for(int r=0;r<3;r++)
					k_points_list(r,counting)+=shift(r);
				counting = counting + 1;
			}
			else
				///to avoid the reading of blank rows			
				break;
		}
		k_points_list_file.close();
		if(crystal_coordinates==1){
			arma::vec k_points_list_tmp(3);
			for(int i=0;i<number_k_points_list;i++){
				for(int s=0;s<3;s++){
					k_points_list_tmp(s)=0.0;
					for(int r=0;r<3;r++)
						k_points_list_tmp(s)+=k_points_list(r,i)*primitive_vectors(s,r);
				}
				for(int s=0;s<3;s++)
					k_points_list(s,i)=k_points_list_tmp(s);
			}
		}
	}else{
		std::random_device rd;
    	std::mt19937 gen(rd());
   		std::uniform_real_distribution<> dis(-1.0,1.0); // distribution in range [-1,1]
		for(int i=0;i<number_k_points_list;i++){
			for(int r=0;r<3;r++)
				k_points_list(r,i)=dis(gen)+shift(r);
		}
		if(crystal_coordinates==1){
			arma::vec k_points_list_tmp(3);
			for(int i=0;i<number_k_points_list;i++){
				for(int s=0;s<3;s++){
					k_points_list_tmp(s)=0.0;
					for(int r=0;r<3;r++)
						k_points_list_tmp(s)+=k_points_list(r,i)*primitive_vectors(s,r);
				}
				for(int s=0;s<3;s++)
					k_points_list(s,i)=k_points_list_tmp(s);
			}
		}
	}
	for (int i = 0; i < number_k_points_list; i++)
		for (int j = 0; j < number_k_points_list; j++)
			for (int r = 0; r < 3; r++)
				k_point_differences(r,i*number_k_points_list+j)=k_points_list(r,i)-k_points_list(r,j);
};
//void K_points::push_k_points_list_values(double spacing_tmp,int dimension_tmp,arma::vec direction_cutting_tmp){
//	dimension=dimension_tmp;
//	direction_cutting=direction_cutting_tmp;
//	spacing=spacing_tmp;
//	if(dimension==3){
//		arma::vec vec_number_k_points_list(3);
//		for (int i = 0; i < 3; i++)
//			vec_number_k_points_list(i) = int(sqrt(accu(primitive_vectors.col(i) % primitive_vectors.col(i))) / spacing);
//		int limiti = int(vec_number_k_points_list(0));
//		int limitj = int(vec_number_k_points_list(1));
//		int limitk = int(vec_number_k_points_list(2));
//		number_k_points_list = limiti * limitj * limitk;
//		k_points_list.set_size(3,number_k_points_list);
//		int counting = 0;
//		for (int i = 0; i < limiti; i++)
//			for (int j = 0; j < limitj; j++)
//				for (int k = 0; k < limitk; k++){
//					for (int r = 0; r < 3; r++)
//						k_points_list(r, counting) = ((double)i / limiti) * (shift(r) + primitive_vectors(r, 0)) + ((double)j / limitj) * (shift(r) + primitive_vectors(r, 1)) + ((double)k / limitk) * (shift(r) + primitive_vectors(r, 2));
//					counting = counting + 1;
//				}
//	}
//	k_point_differences.set_size(3,number_k_points_list*number_k_points_list);
//	for (int i = 0; i < number_k_points_list; i++)
//		for (int j = 0; j < number_k_points_list; j++)
//			for (int r = 0; r < 3; r++)
//				k_point_differences(r,i*number_k_points_list+j)=k_points_list(r,i)-k_points_list(r,j);
//	///TO IMPLEMENT OTHER CONDITIONS
//};
arma::vec K_points::pull_shift(){
	return shift;
};
arma::mat K_points::pull_k_points_list_values(){
	return k_points_list;
};
int K_points::pull_number_k_points_list(){
	return number_k_points_list;
};
void K_points::print(){
	cout << "K points list" << endl;
	cout << number_k_points_list <<endl;
	for (int i = 0; i < number_k_points_list; i++){
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
	arma::vec number_g_points_direction;
	double** g_points_list;
	double cutoff_g_points_list;
	int dimension_g_points_list;
	arma::vec direction_cutting{arma::vec(3)};
	arma::mat bravais_lattice{arma::mat(3,3)};
	arma::vec shift{arma::vec(3)};
public:
	G_points(Crystal_Lattice *crystal_lattice,double cutoff_g_points_list_tmp,int dimension_g_points_list_tmp,arma::vec direction_cutting_tmp,arma::vec shift_tmp);
	arma::mat pull_g_points_list_values();
	int pull_number_g_points_list();
	void print();
	~G_points(){
		number_g_points_list=0;
		cutoff_g_points_list=0.0;
		dimension_g_points_list=0;
		for(int r=0;r<3;r++)
			delete[] g_points_list[r];
		delete[] g_points_list;
	};
};
G_points::G_points(Crystal_Lattice *crystal_lattice,double cutoff_g_points_list_tmp,int dimension_g_points_list_tmp,arma::vec direction_cutting_tmp,arma::vec shift_tmp){
	dimension_g_points_list=dimension_g_points_list_tmp;
	direction_cutting=direction_cutting_tmp;
	cutoff_g_points_list=cutoff_g_points_list_tmp;
	shift=shift_tmp;

	arma::mat primitive_vectors=crystal_lattice->pull_primitive_vectors();
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
			g_points_list=new double*[3];
			for(int s=0;s<3;s++)
				g_points_list[s]=new double[number_g_points_list];
			int counting=0;	
			for (int i = -number_g_points_direction(0); i <= number_g_points_direction(0); i++)
				for (int j = -number_g_points_direction(1); j <= number_g_points_direction(1); j++)
					for (int k = -number_g_points_direction(2); k <= number_g_points_direction(2); k++){
							for (int r = 0; r < 3; r++)
								g_points_list[r][counting] =  i * (shift(r) + primitive_vectors(r, 0)) + j * (shift(r) + primitive_vectors(r, 1)) + k * (shift(r) + primitive_vectors(r, 2));
							counting = counting + 1;
					}
		}else if(dimension_g_points_list==2){
			number_g_points_direction.zeros(2);
			arma::mat reciprocal_plane_along; reciprocal_plane_along.zeros(3,2);
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
			g_points_list=new double*[3];
			for(int s=0;s<3;s++)
				g_points_list[s]=new double[number_g_points_list];
			counting = 1;
			for (int i = -number_g_points_direction(0); i <= number_g_points_direction(0); i++)
				for (int j = -number_g_points_direction(1); j <= number_g_points_direction(1); j++){
					//if((i!=0)||(j!=0)){
						for (int r = 0; r < 3; r++)
							g_points_list[r][counting] = i * (shift(r) + reciprocal_plane_along(r, 0)) + j * (shift(r) + reciprocal_plane_along(r, 1));
						//cout<<g_points_list.col(count)<<endl;
						counting = counting + 1;
					//}else
					//	for (int r = 0; r < 3; r++)
					//		g_points_list(r, 0) = 0.0;
				}
		}else{
		/// TO IMPLEMENT OTHER CASE
		cout<<"TO IMPLEMENT"<<endl;
		}
	}
};
arma::mat G_points::pull_g_points_list_values(){
	arma::mat g_points_list_new(3,number_g_points_list);
	for(int r=0;r<3;r++)
		for(int s=0;s<number_g_points_list;s++)
			g_points_list_new(r,s)=g_points_list[r][s];

	return g_points_list_new;
};
int G_points::pull_number_g_points_list(){
	return number_g_points_list;
};
void G_points::print(){
	cout << "G points list "<<number_g_points_list << endl;
	for (int i = 0; i < number_g_points_list; i++){
		cout << " ( ";
		for (int r = 0; r < 3; r++)
			cout << i<<" "<< g_points_list[r, i] << " ";
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
	arma::vec weights_primitive_cells;
	arma::mat positions_primitive_cells;
	arma::field<arma::cx_cube> hamiltonian;
	arma::field<arma::mat> wannier_centers;
	bool dynamic_shifting;
	double fermi_energy;
	double little_shift;
	double scissor_operator;
	arma::mat bravais_lattice{arma::mat(3,3)};
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
	Hamiltonian_TB(string wannier90_hr_file_name,string wannier90_centers_file_name,double fermi_energy_tmp,int spinorial_calculation_tmp,int number_atoms_tmp,bool dynamic_shifting_tmp,double little_shift_tmp,double scissor_operator_tmp,arma::mat bravais_lattice_tmp,int number_primitive_cells_tmp,int number_wannier_functions_tmp);
	arma::field<arma::cx_mat> FFT(arma::vec k_point);
	std::tuple<arma::mat,arma::cx_mat> pull_ks_states(arma::vec k_point);
	std::tuple<arma::mat,arma::cx_mat> pull_ks_states_subset(arma::vec k_point,int number_valence_bands_selected,int number_conduction_bands_selected);
	arma::field<arma::cx_cube> pull_hamiltonian();
	int pull_htb_basis_dimension();
	int pull_number_wannier_functions();
	double pull_fermi_energy();
	void print_hamiltonian();
	void print_ks_states(arma::vec k_point, int number_valence_bands_selected, int number_conduction_bands_selected);
	arma::field<arma::mat> pull_wannier_centers();
	arma::mat pull_bravais_lattice(){
		return bravais_lattice;
	};
	void pull_bands(string bands_file_name,string k_points_bands_file_name,int number_k_points_bands, int number_valence_bands_selected,int number_conduction_bands_selected, int crystal_coordinates,arma::mat primitive_vectors);
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
Hamiltonian_TB::Hamiltonian_TB(string wannier90_hr_file_name,string wannier90_centers_file_name,double fermi_energy_tmp,int spinorial_calculation_tmp,int number_atoms_tmp,bool dynamic_shifting_tmp,double little_shift_tmp,double scissor_operator_tmp,arma::mat bravais_lattice_tmp,int number_primitive_cells_tmp,int number_wannier_functions_tmp):
weights_primitive_cells(number_primitive_cells_tmp), positions_primitive_cells(3,number_primitive_cells_tmp)
{
	cout<<"Be Carefull: if you are doing a collinear spin calculation, the number of Wannier functions in the two spin channels has to be the same!!"<<endl;
	fermi_energy=fermi_energy_tmp;
	number_atoms=number_atoms_tmp;
	number_wannier_functions=number_wannier_functions_tmp;
	number_primitive_cells=number_primitive_cells_tmp;
	spinorial_calculation=spinorial_calculation_tmp;
	dynamic_shifting=dynamic_shifting_tmp;
	scissor_operator=scissor_operator_tmp;
	bravais_lattice=bravais_lattice_tmp;
	ifstream wannier90_hr_file;
	ifstream wannier90_centers_file;
	wannier90_hr_file.open(wannier90_hr_file_name);
	wannier90_centers_file.open(wannier90_centers_file_name);
	int number_primitive_cells_check;
	int number_wannier_functions_check;
	int counting_primitive_cells_check;
	hamiltonian.set_size(spinorial_calculation+1);
	wannier_centers.set_size(spinorial_calculation+1);

	cout<<"Reading Hamiltonian..."<<endl;
	int total_elements;
	string history_time;
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
		wannier90_hr_file >> number_wannier_functions_check;
		wannier90_hr_file >> number_primitive_cells_check;
		if((number_wannier_functions_check!=number_wannier_functions)||(number_primitive_cells_check!=number_primitive_cells)){
			cout<<"ERROR!!!!!!!"<<endl;
		}
		cout<<"Number wannier functions "<<number_wannier_functions<<endl;
		cout<<"Number primitive cells "<<number_primitive_cells<<endl;
		if (spin_channel == 0){
			// initialization of the vriables
			if (spinorial_calculation == 1){
				/// two loops
				//hamiltonian.set_size(2);
				hamiltonian(0).set_size(number_wannier_functions, number_wannier_functions, number_primitive_cells);
				hamiltonian(1).set_size(number_wannier_functions, number_wannier_functions, number_primitive_cells);
				htb_basis_dimension = number_wannier_functions * 2;
			}
			else{
				/// one loop
				///hamiltonian.set_size(1);
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
		counting_primitive_cells_check = 0;
		counting_positions = 0;
		/// the hamiltonian in the collinear case is diagonal in the spin channel
		while (counting_positions < total_elements){
			if (counting_positions == number_wannier_functions * number_wannier_functions * counting_primitive_cells_check)
			{
				wannier90_hr_file >> positions_primitive_cells(0, counting_primitive_cells_check) >> positions_primitive_cells(1, counting_primitive_cells_check) >> positions_primitive_cells(2, counting_primitive_cells_check);	
				counting_primitive_cells_check = counting_primitive_cells_check + 1;
			}
			else
				wannier90_hr_file >> trashing_positions[0] >> trashing_positions[1] >> trashing_positions[2];
			wannier90_hr_file >> l >> m;
			wannier90_hr_file >> real_part >> imag_part;

			hamiltonian(spin_channel)(l - 1, m - 1, counting_primitive_cells_check - 1).real(real_part);
			hamiltonian(spin_channel)(l - 1, m - 1, counting_primitive_cells_check - 1).imag(imag_part);
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
	arma::vec position_primitive_cells_tmp(3);
	for(int i=0;i<number_primitive_cells;i++){
		for(int s=0;s<3;s++){
			position_primitive_cells_tmp(s)=0.0;
			for(int r=0;r<3;r++)
				position_primitive_cells_tmp(s)+=positions_primitive_cells(r,i)*bravais_lattice(s,r);
		}
		for(int s=0;s<3;s++)
			positions_primitive_cells(s,i)=position_primitive_cells_tmp(s);
	}
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
				//wannier_centers.set_size(2);
				wannier_centers(0).set_size(3,number_wannier_functions);
				wannier_centers(1).set_size(3,number_wannier_functions);
			}
			else{
				/// one loop
				//wannier_centers.set_size(1);
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
arma::field<arma::cx_cube> Hamiltonian_TB::pull_hamiltonian(){
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
arma::field<arma::mat> Hamiltonian_TB::pull_wannier_centers(){
	return wannier_centers;
};
arma::field<arma::cx_mat> Hamiltonian_TB::FFT(arma::vec k_point){
	int flag_spin_channel = 0;
	int offset;
	
	arma::vec temporary_cos(number_primitive_cells);
	arma::vec temporary_sin(number_primitive_cells);
	arma::vec real_part_hamiltonian(number_primitive_cells);
	arma::vec imag_part_hamiltonian(number_primitive_cells);
	double variable_tmp;
	//#pragma omp parallel for 
	for (int r = 0; r < number_primitive_cells; r++){
		variable_tmp=0.0;
		for(int s = 0; s < 3; s++)
			variable_tmp+=k_point(s)*positions_primitive_cells(s,r);
		temporary_cos(r) = std::cos(variable_tmp);
		temporary_sin(r) = std::sin(variable_tmp);
		///cout<<r<<positions_primitive_cells.col(r)<<" "<<k_point<<" "<<temporary_cos(r)<<" "<<temporary_sin(r)<<endl;
	}
	if (spinorial_calculation == 1){
		arma::field<arma::cx_mat> fft_hamiltonian(2);
		fft_hamiltonian(0).zeros(number_wannier_functions,number_wannier_functions);
		fft_hamiltonian(1).zeros(number_wannier_functions,number_wannier_functions);
		while (flag_spin_channel < 2){
			offset = number_wannier_functions * flag_spin_channel;
			//#pragma omp parallel for collapse(2) private(real_part_hamiltonian,imag_part_hamiltonian)
			for (int l = 0; l < number_wannier_functions; l++)
				for (int m = 0; m < number_wannier_functions; m++){
					real_part_hamiltonian = arma::real(hamiltonian(flag_spin_channel).tube(l, m));
					imag_part_hamiltonian = arma::imag(hamiltonian(flag_spin_channel).tube(l, m));
					real_part_hamiltonian = real_part_hamiltonian%weights_primitive_cells;
					imag_part_hamiltonian = imag_part_hamiltonian%weights_primitive_cells;
					fft_hamiltonian(flag_spin_channel)(l, m).real(arma::accu(real_part_hamiltonian % temporary_cos) - arma::accu(imag_part_hamiltonian % temporary_sin));
					fft_hamiltonian(flag_spin_channel)(l, m).imag(arma::accu(imag_part_hamiltonian % temporary_cos) + arma::accu(real_part_hamiltonian % temporary_sin));
				}
			//cout<<flag_spin_channel<<" "<<fft_hamiltonian(flag_spin_channel)<<endl;
			flag_spin_channel++;
		}
		return fft_hamiltonian;
	}else{
		arma::field<arma::cx_mat> fft_hamiltonian(1);
		fft_hamiltonian(0).zeros(number_wannier_functions, number_wannier_functions);
		//#pragma omp parallel for collapse(2) private(real_part_hamiltonian,imag_part_hamiltonian)
		for (int l = 0; l < number_wannier_functions; l++)
		{
			for (int m = 0; m < number_wannier_functions; m++)
			{
				///cout<<l<<m<<endl;
				real_part_hamiltonian = arma::real(hamiltonian(0).tube(l, m));
				imag_part_hamiltonian = arma::imag(hamiltonian(0).tube(l, m));
				real_part_hamiltonian = real_part_hamiltonian%weights_primitive_cells;
				imag_part_hamiltonian = imag_part_hamiltonian%weights_primitive_cells;
				fft_hamiltonian(0)(l, m).real(arma::accu(real_part_hamiltonian % temporary_cos) - arma::accu(imag_part_hamiltonian % temporary_sin));
				fft_hamiltonian(0)(l, m).imag(arma::accu(imag_part_hamiltonian % temporary_cos) + arma::accu(real_part_hamiltonian % temporary_sin));
			}
		}
		//cout<<"end fft"<<endl;
		return fft_hamiltonian;
	}	
};
std::tuple<arma::mat, arma::cx_mat> Hamiltonian_TB::pull_ks_states(arma::vec k_point){
	/// the eigenvalues are saved into a two component element, in order to make the code more general
	arma::mat ks_eigenvalues_spinor(2, number_wannier_functions, arma::fill::zeros);
	arma::cx_mat ks_eigenvectors_spinor(htb_basis_dimension, number_wannier_functions, arma::fill::zeros);

	////field<cx_mat> fft_hamiltonian = FFT(k_point);
	if (spinorial_calculation == 1){
		arma::cx_mat hamiltonian_up(number_wannier_functions, number_wannier_functions);
		arma::cx_mat hamiltonian_down(number_wannier_functions, number_wannier_functions);
		arma::field<arma::cx_mat> fft_hamiltonian = FFT(k_point);
		hamiltonian_up = fft_hamiltonian(0);
		hamiltonian_down = fft_hamiltonian(1);
		//cout<<"diagonanlization starting"<<endl;
		arma::vec eigenvalues_up;
		arma::cx_mat eigenvectors_up;
		arma::vec eigenvalues_down;
		arma::cx_mat eigenvectors_down;
		arma::eig_sym(eigenvalues_up,eigenvectors_up,hamiltonian_up);
		arma::eig_sym(eigenvalues_down,eigenvectors_down,hamiltonian_down);

		arma::uvec ordering_up = arma::sort_index(eigenvalues_up);
		arma::uvec ordering_down = arma::sort_index(eigenvalues_down);
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
		arma::field<arma::cx_mat> fft_hamiltonian = FFT(k_point);
		
		arma::vec eigenvalues;
		arma::cx_mat eigenvectors;
		arma::eig_sym(eigenvalues,eigenvectors,fft_hamiltonian(0));

		arma::uvec ordering = arma::sort_index(eigenvalues);

		/// in the case of spinorial_calculation=1 combining the two components of spin into a single spinor
		/// saving the ordered eigenvectors in the matrix ks_eigenvectors_spinor
		///#pragma omp parallel for collapse(2)
		for (int i = 0; i < number_wannier_functions; i++)
			for (int j = 0; j < htb_basis_dimension; j++){
				ks_eigenvectors_spinor(j, i) = eigenvectors(j, ordering(i))/arma::vecnorm(eigenvectors.col(ordering(i)));
			}

		for (int i = 0; i < number_wannier_functions; i++){
			ks_eigenvalues_spinor(0,i)=real(eigenvalues(ordering(i)));
			ks_eigenvalues_spinor(1,i)=real(eigenvalues(ordering(i)));
		}

		return {ks_eigenvalues_spinor, ks_eigenvectors_spinor};
	}
};
std::tuple<arma::mat,arma::cx_mat> Hamiltonian_TB::pull_ks_states_subset(arma::vec k_point,int number_valence_bands_selected,int number_conduction_bands_selected){
	int number_valence_bands = 0;
	int number_conduction_bands = 0;
	int dimensions_subspace = number_conduction_bands_selected + number_valence_bands_selected;
	arma::vec spinor_scissor_operator(2);
	spinor_scissor_operator(0)=scissor_operator;
	spinor_scissor_operator(1)=scissor_operator;
	
	arma::mat ks_eigenvalues(2,number_wannier_functions, arma::fill::zeros);
	arma::cx_mat ks_eigenvectors(htb_basis_dimension, number_wannier_functions, arma::fill::zeros);
	std::tuple<arma::mat,arma::cx_mat> ks_states(ks_eigenvalues,ks_eigenvectors);
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
	//cout<<"Number valence bands "<<number_valence_bands<<" Number conduction bands "<<number_conduction_bands<<endl;
	
	/// in a single matrix: first are written valence states, than (at higher rows) conduction states
	/// converting from eV to Ang^{-1}	
	arma::mat ks_eigenvalues_subset(2, dimensions_subspace);
	arma::cx_mat ks_eigenvectors_subset(htb_basis_dimension, dimensions_subspace);
	for (int i = 0; i < dimensions_subspace; i++){
		///cout<<ks_eigenvalues.col(i)<<endl;
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

void Hamiltonian_TB:: pull_bands(string bands_file_name,string k_points_bands_file_name, int number_k_points_bands,int number_valence_bands_selected,int number_conduction_bands_selected, int crystal_coordinates,arma::mat primitive_vectors){
	arma::mat k_point(3,2);
	arma::mat k_point_tmp(3,2);
	arma::vec k_point_3(3);
	int intermediate_points;
	arma::mat eigenvalues(2,number_conduction_bands_selected+number_valence_bands_selected);
	ifstream k_points_bands_file;
	ofstream bands_file;
	k_points_bands_file.open(k_points_bands_file_name);
	bands_file.open(bands_file_name);
	k_points_bands_file.seekg(0);
	int counting = 0;
	while (k_points_bands_file.peek()!=EOF){
		if (counting<number_k_points_bands)
		{
			k_points_bands_file >> k_point(0,0) >> k_point(1,0) >> k_point(2,0);
			k_points_bands_file >> k_point(0,1) >> k_point(1,1) >> k_point(2,1);
			k_points_bands_file >> intermediate_points;
			counting = counting + 1;
			cout<<k_point<<endl;
			if(crystal_coordinates==1){
				for(int k=0;k<2;k++){
					for(int s=0;s<3;s++){
						k_point_tmp(s,k)=0.0;
						for(int r=0;r<3;r++)
							k_point_tmp(s,k)+=k_point(r,k)*primitive_vectors(s,r);
					}
					for(int s=0;s<3;s++)
						k_point(s,k)=k_point_tmp(s,k);
				}
			}
			for(int j=0;j<intermediate_points;j++){
				for(int r=0;r<3;r++){
					k_point_3(r)=k_point(r,0)*(1.0-double(j)/double(intermediate_points-1))+k_point(r,1)*(double(j)/double(intermediate_points-1));
					//cout<<k_point_3(r)<<" ";
				}
				//cout<<endl;
				///cout<<k_point_3<<endl;
				///cout<<double(j)/double(intermediate_points)<<endl;
				eigenvalues=get<0>(pull_ks_states_subset(k_point_3,number_valence_bands_selected,number_conduction_bands_selected));
				for(int spin=0;spin<(spinorial_calculation+1);spin++)
					for(int i=0;i<number_conduction_bands_selected+number_valence_bands_selected;i++)
						bands_file<<eigenvalues(spin,i)<<" ";
				bands_file<<endl;
			}
		}
		else
			///to avoid the reading of blank rows			
			break;
	}
	k_points_bands_file.close();
	bands_file.close();
};
	

void Hamiltonian_TB::print_ks_states(arma::vec k_point, int number_valence_bands_selected, int number_conduction_bands_selected){
	std::tuple<arma::mat,arma::cx_mat> results_htb;
	std::tuple<arma::mat,arma::cx_mat> results_htb_subset;
	cout<<"Extraction all ks states:"<<endl;
	results_htb=pull_ks_states(k_point);
	cout<<"Extraction subset ks states"<<endl;
	results_htb_subset=pull_ks_states_subset(k_point,number_valence_bands_selected,number_conduction_bands_selected);
	arma::mat eigenvalues=get<0>(results_htb);
	arma::cx_mat eigenvectors=get<1>(results_htb);
	arma::mat eigenvalues_subset=get<0>(results_htb_subset);
	arma::cx_mat eigenvectors_subset=get<1>(results_htb_subset);
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

class Real_space_wannier
{
private:
 	int number_wannier_functions;
	arma::vec number_points_real_space_grid{arma::vec(3)};
	arma::vec number_unit_cells_supercell{arma::vec(3)};
	arma::vec origin{arma::vec(3)};
	arma::mat supercell_axis{arma::mat(3,3)};
	int spinorial_calculation;
	int number_points_real_space_grid_total;
	arma::vec number_points_real_space_grid_percell{arma::vec(3)};
	arma::vec origin_unitcell{arma::vec(3)};
	string seedname_files_xsf;
	double cell_volume;
	arma::mat real_space_wannier_functions_list;
	arma::mat atoms_coordinates;
	int number_atoms;
public:
	Real_space_wannier(arma::vec number_points_real_space_grid_tmp, arma::vec number_unit_cells_supercell_tmp, int spinorial_calculation_tmp, string seedname_files_xsf_tmp, int number_wannier_functions_tmp, double cell_volume_tmp, int number_atoms_tmp, arma::mat atoms_coordinates_tmp);
	arma::mat pull_real_space_wannier_functions_list(){
		return real_space_wannier_functions_list;
	};
	arma::vec pull_origin(){
		return origin;
	};
	arma::vec pull_origin_unitcell(){
		return origin_unitcell;
	};
	arma::mat pull_supercell_axis(){
		return supercell_axis;
	};
	arma::vec pull_number_points_real_space_grid(){
		return number_points_real_space_grid;
	};
	arma::vec pull_number_points_real_space_grid_percell(){
		return number_points_real_space_grid_percell;
	};
	arma::vec pull_number_unit_cells_supercell(){
		return number_unit_cells_supercell;
	};
	int pull_number_wannier_functions(){
		return number_wannier_functions;
	};
	void print(int which_wannier_function, arma::vec which_cell, int all_cells, double isovalue_pos, double isovalue_neg,string wannier_file_name){
		cout<<number_wannier_functions<<endl;
		cout<<number_points_real_space_grid_percell<<endl;
		cout<<number_unit_cells_supercell<<endl;
		cout<<number_points_real_space_grid<<endl;
		cout<<origin<<endl;
		cout<<supercell_axis<<endl;
		cout<<supercell_axis.col(0)<<endl;
		int count=0;
		double temporary;

		ofstream wannier_file;
		wannier_file.open(wannier_file_name);
		///double max_value=0.0;
		///double min_value=1000.0;
		
		wannier_file<<"CRYSTAL"<<endl;
		wannier_file<<"PRIMVEC"<<endl;
		for(int r=0;r<3;r++)
			wannier_file<<(supercell_axis.col(r)).t();
		wannier_file<<"CONVEC"<<endl;
		for(int r=0;r<3;r++)
			wannier_file<<(supercell_axis.col(r)).t();
		wannier_file<<"PRIMCOORD"<<endl;
		wannier_file<<number_atoms<<"  1 "<<endl;
		for(int r=0;r<number_atoms;r++)
			wannier_file<<"X" <<" "<<(atoms_coordinates.col(r)).t();
		wannier_file<<"BEGIN_BLOCK_DATAGRID_3D"<<endl;
		wannier_file<<"3D_field"<<endl;
		wannier_file<<"BEGIN_DATAGRID_3D_UNKNOWN"<<endl;
		if(all_cells==1){
			for(int r=0;r<3;r++)
				wannier_file<<int(number_points_real_space_grid(r))<<" ";
			wannier_file<<endl;
			wannier_file<<origin.t();
			for(int r=0;r<3;r++)
				wannier_file<<(supercell_axis.col(r)*number_unit_cells_supercell(r)).t();
			for(int spin=0;spin<(spinorial_calculation+1);spin++)
				for(int i=0;i<number_unit_cells_supercell(0);i++)
					for(int j=0;j<number_unit_cells_supercell(1);j++)
						for(int k=0;k<number_unit_cells_supercell(2);k++)
							for(int s=0;s<number_points_real_space_grid_percell(0);s++)
								for(int t=0;t<number_points_real_space_grid_percell(1);t++)
									for(int l=0;l<number_points_real_space_grid_percell(2);l++){
										temporary=real_space_wannier_functions_list(s*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t*number_points_real_space_grid_percell(2)+l,spin*number_wannier_functions*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+which_wannier_function*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j*number_unit_cells_supercell(2)+k);
										///cout<<temporary<<endl;
										///if(temporary>max_value){
										///	max_value=temporary;
										///	///cout<<max_value;
										///	///cout<<temporary;
										///}
										///if(temporary<min_value)
										///	min_value=temporary;
										///if(temporary>isovalue_neg&&temporary<isovalue_pos){
										///for(int r=0;r<3;r++)
										///	wannier_file<<origin(r)+(i+s/number_points_real_space_grid_percell(0))*supercell_axis(r,0)+(j+t/number_points_real_space_grid_percell(1))*supercell_axis(r,1)+(k+l/number_points_real_space_grid_percell(2))*supercell_axis(r,2)<<" ";
										wannier_file<<temporary<<"  ";
										///}else{
										///	for(int r=0;r<3;r++)
										///		wannier_file<<origin(r)+(i+s/number_points_real_space_grid_percell(0))*supercell_axis(r,0)+(j+t/number_points_real_space_grid_percell(1))*supercell_axis(r,1)+(k+l/number_points_real_space_grid_percell(2))*supercell_axis(r,2)<<" ";
										///	wannier_file<<0.0<<endl;
										///}
										if(count==5){
											wannier_file<<endl;
											count=0;
										}else{
											count+=1;
										}
									}
			cout<<"central cell"<<endl;
			cout<<number_unit_cells_supercell<<endl;
			int i=origin_unitcell(0);
			int j=origin_unitcell(1);
			int k=origin_unitcell(2);
			cout<<i<<j<<k<<endl;
			int s=0; int t=0; int l=0;
			int spin=0;
			for(int i=0;i<number_unit_cells_supercell(0);i++)
				for(int j=0;j<number_unit_cells_supercell(1);j++)
					for(int k=0;k<number_unit_cells_supercell(2);k++){
						for(int r=0;r<3;r++)
							cout<<origin(r)+(i+s/number_points_real_space_grid_percell(0))*supercell_axis(r,0)+(j+t/number_points_real_space_grid_percell(1))*supercell_axis(r,1)+(k+l/number_points_real_space_grid_percell(2))*supercell_axis(r,2)<<" ";
						cout<<real_space_wannier_functions_list(s*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t*number_points_real_space_grid_percell(2)+l,spin*number_wannier_functions*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+which_wannier_function*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j*number_unit_cells_supercell(2)+k)<<endl;
					}
			wannier_file<<"END_DATAGRID_3D"<<endl;
			wannier_file<<"END_BLOCK_DATAGRID_3D"<<endl;
		}else{
			int i=which_cell(0);
			int j=which_cell(1);
			int k=which_cell(2);
			for(int r=0;r<3;r++)
				wannier_file<<int(number_points_real_space_grid_percell(r))<<" ";
			wannier_file<<endl;
			for(int r=0;r<3;r++)
				wannier_file<<origin(r)+i*supercell_axis(r,0)+j*supercell_axis(r,1)+k*supercell_axis(r,2)<<" ";
			wannier_file<<endl;
			for(int r=0;r<3;r++)
				wannier_file<<(supercell_axis.col(r)).t();
			for(int spin=0;spin<(spinorial_calculation+1);spin++)
				for(int s=0;s<number_points_real_space_grid_percell(0);s++)
					for(int t=0;t<number_points_real_space_grid_percell(1);t++)
						for(int l=0;l<number_points_real_space_grid_percell(2);l++){
							temporary=real_space_wannier_functions_list(s*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t*number_points_real_space_grid_percell(2)+l,spin*number_wannier_functions*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+which_wannier_function*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j*number_unit_cells_supercell(2)+k);
							///if(temporary>max_value)
							///	max_value=temporary;
							///if(temporary<min_value)
							///	min_value=temporary;
							///if(temporary>isovalue_neg&&temporary<isovalue_pos){
							///for(int r=0;r<3;r++)
							///	wannier_file<<origin(r)+(i+s/number_points_real_space_grid_percell(0))*supercell_axis(r,0)+(j+t/number_points_real_space_grid_percell(1))*supercell_axis(r,1)+(k+l/number_points_real_space_grid_percell(2))*supercell_axis(r,2)<<" ";
							wannier_file<<temporary<<" ";
							///}else{
							///	for(int r=0;r<3;r++)
							///		wannier_file<<origin(r)+(i+s/number_points_real_space_grid_percell(0))*supercell_axis(r,0)+(j+t/number_points_real_space_grid_percell(1))*supercell_axis(r,1)+(k+l/number_points_real_space_grid_percell(2))*supercell_axis(r,2)<<" ";
							///	wannier_file<<0.0<<endl;
							///}
							if(count==5){
										wannier_file<<endl;
										count=0;
									}else{
										count+=1;
									}
						}
		wannier_file<<"END_DATAGRID_3D"<<endl;
		wannier_file<<"END_BLOCK_DATAGRID_3D"<<endl;
		}
		///cout<<"max min "<<max_value<<" "<<min_value<<endl;
	};
};

Real_space_wannier::Real_space_wannier(arma::vec number_points_real_space_grid_tmp, arma::vec number_unit_cells_supercell_tmp, int spinorial_calculation_tmp, string seedname_files_xsf_tmp, int number_wannier_functions_tmp,double cell_volume_tmp,int number_atoms_tmp, arma::mat atoms_coordinates_tmp):
real_space_wannier_functions_list(number_points_real_space_grid_tmp(0)*number_points_real_space_grid_tmp(1)*number_points_real_space_grid_tmp(2)/(number_unit_cells_supercell_tmp(0)*number_unit_cells_supercell_tmp(1)*number_unit_cells_supercell_tmp(2)),(spinorial_calculation_tmp+1)*number_wannier_functions_tmp*number_unit_cells_supercell_tmp(0)*number_unit_cells_supercell_tmp(1)*number_unit_cells_supercell_tmp(2))
{
	number_unit_cells_supercell=number_unit_cells_supercell_tmp;
	spinorial_calculation=spinorial_calculation_tmp;
	seedname_files_xsf=seedname_files_xsf_tmp;
	number_points_real_space_grid=number_points_real_space_grid_tmp;
	number_points_real_space_grid_total=number_points_real_space_grid(0)*number_points_real_space_grid(1)*number_points_real_space_grid(2);
	number_wannier_functions=number_wannier_functions_tmp;

	atoms_coordinates=atoms_coordinates_tmp;
	number_atoms=number_atoms_tmp;
	arma::vec number_points_real_space_grid_test(3);	
	for(int i=0;i<3;i++)
		number_points_real_space_grid_percell(i)=int(number_points_real_space_grid(i)/number_unit_cells_supercell(i));
	
	double normalize;

	cell_volume=cell_volume_tmp;

	string file_name;
	string useless_lines;
	string extension_file_names=".xsf";
	ifstream  wannier_file_xsf;

	cout<<size(real_space_wannier_functions_list)<<endl;
	for(int spin=0;spin<(spinorial_calculation+1);spin++)
		for(int w=0;w<number_wannier_functions;w++){
			stringstream number;
			number<<setw(5) <<setfill('0')<<w+1;
			if(spinorial_calculation==0)
				file_name=seedname_files_xsf+"_"+number.str()+extension_file_names;
			else{
				if(spin==0)
					file_name=seedname_files_xsf+"up_"+number.str()+extension_file_names;
				else
					file_name=seedname_files_xsf+"down_"+number.str()+extension_file_names;
			}
			number.str(string());
			cout<<file_name<<endl;
			wannier_file_xsf.open(file_name);
			wannier_file_xsf.seekg(0);
			while (wannier_file_xsf.peek() != EOF){
				///useless lines
				for(int s=0;s<22;s++){
					getline(wannier_file_xsf, useless_lines);
				}
				wannier_file_xsf >> number_points_real_space_grid_test(0) >> number_points_real_space_grid_test(1) >> number_points_real_space_grid_test(2);
				wannier_file_xsf >> origin(0) >> origin(1) >> origin(2);
				for(int p=0;p<3;p++)
					wannier_file_xsf >> supercell_axis(0,p) >> supercell_axis(1,p) >> supercell_axis(2,p);
				///reading of the wannier function
				normalize=0;
				///#pragma omp parallel collapse(6) private(spin)
				///{
				for(int k=0;k<number_unit_cells_supercell(2);k++)	
					for(int l=0;l<number_points_real_space_grid_percell(2);l++)
						for(int j=0;j<number_unit_cells_supercell(1);j++)
							for(int t=0;t<number_points_real_space_grid_percell(1);t++)
								for(int i=0;i<number_unit_cells_supercell(0);i++)
									for(int s=0;s<number_points_real_space_grid_percell(0);s++){
										wannier_file_xsf>>real_space_wannier_functions_list(s*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t*number_points_real_space_grid_percell(2)+l,spin*number_wannier_functions*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+w*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j*number_unit_cells_supercell(2)+k);
										normalize=normalize+((real_space_wannier_functions_list(s*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t*number_points_real_space_grid_percell(2)+l,spin*number_wannier_functions*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+w*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j*number_unit_cells_supercell(2)+k))*real_space_wannier_functions_list(s*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t*number_points_real_space_grid_percell(2)+l,spin*number_wannier_functions*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+w*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j*number_unit_cells_supercell(2)+k));	
									}
				///}

			}
			wannier_file_xsf.close();
			normalize=normalize/(double(number_points_real_space_grid_total));

			double testing_normalization=0.0;
			for(int i=0;i<number_unit_cells_supercell(0);i++)
				for(int j=0;j<number_unit_cells_supercell(1);j++)
					for(int k=0;k<number_unit_cells_supercell(2);k++)
						for(int s=0;s<number_points_real_space_grid_percell(0);s++)
							for(int t=0;t<number_points_real_space_grid_percell(1);t++)
								for(int l=0;l<number_points_real_space_grid_percell(2);l++){
									real_space_wannier_functions_list(s*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t*number_points_real_space_grid_percell(2)+l,spin*number_wannier_functions*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+w*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j*number_unit_cells_supercell(2)+k)=real_space_wannier_functions_list(s*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t*number_points_real_space_grid_percell(2)+l,spin*number_wannier_functions*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+w*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j*number_unit_cells_supercell(2)+k)/sqrt(normalize);
									testing_normalization=testing_normalization+(real_space_wannier_functions_list(s*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t*number_points_real_space_grid_percell(2)+l,spin*number_wannier_functions*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+w*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j*number_unit_cells_supercell(2)+k))*real_space_wannier_functions_list(s*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t*number_points_real_space_grid_percell(2)+l,spin*number_wannier_functions*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+w*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j*number_unit_cells_supercell(2)+k);
								}
			////cout<<"normalization "<<normalize<<" "<<testing_normalization/double(number_points_real_space_grid_total)<<endl;
		}

		////normalizing spurcell axis
		for(int p=0;p<3;p++)
			for(int r=0;r<3;r++)
				supercell_axis(r,p)=supercell_axis(r,p)/number_unit_cells_supercell(p);

		///finding the unit cell between the different 
		arma::mat different_origins(3,number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2));
		int count=0;
		for(int i=0;i<number_unit_cells_supercell(0);i++)
			for(int j=0;j<number_unit_cells_supercell(1);j++)
				for(int k=0;k<number_unit_cells_supercell(2);k++){
					for(int r=0;r<3;r++)
						different_origins(r,count)=origin(r)+i*supercell_axis(r,0)+j*supercell_axis(r,1)+k*supercell_axis(r,2);
					count+=1;
				}
		double min_value=arma::vecnorm(different_origins.col(0));
		count=0;
		for(int i=0;i<number_unit_cells_supercell(0);i++)
			for(int j=0;j<number_unit_cells_supercell(1);j++)
				for(int k=0;k<number_unit_cells_supercell(2);k++){
					if(arma::vecnorm(different_origins.col(count))<min_value){
						origin_unitcell(0)=i;
						origin_unitcell(1)=j;
						origin_unitcell(2)=k;
						min_value=arma::vecnorm(different_origins.col(count));
					}
					count+=1;
				}

		cout<< "unit cell inside the supercell? distance from lattice:" << min_value<<endl;
};

// Coulomb_Potential class
class Coulomb_Potential
{
private:
	double minimum_k_point_modulus;
	arma::mat primitive_vectors{arma::mat(3,3)};
	arma::vec direction_cutting{arma::vec(3)};
	int dimension_potential;
	K_points *k_points;
	G_points *g_points;
	double volume_cell;
	double radius;
public:
	Coulomb_Potential(K_points* k_points_tmp,G_points* g_points_tmp,double minimum_k_point_modulus_tmp,int dimension_potential_tmp,arma::vec direction_cutting_tmp,double volume_tmp,double radius_tmp);
	double pull(arma::vec k_point);
	double pull_volume();
	void print();
	void print_profile(int number_k_points,double max_radius,string file_coulomb_potential_name, int direction_profile_xyz);
	///~Coulomb_Potential(){
	///	k_points=NULL;
	///	g_points=NULL;
	///};
};
Coulomb_Potential::Coulomb_Potential(K_points* k_points_tmp,G_points* g_points_tmp,double minimum_k_point_modulus_tmp,int dimension_potential_tmp,arma::vec direction_cutting_tmp,double volume_tmp,double radius_tmp){
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
	arma::vec k_point(3,arma::fill::zeros);
	ofstream file_coulomb_potential;
	file_coulomb_potential.open(file_coulomb_potential_name);
	double coulomb;
	for(int i=0;i<number_k_points;i++){
		k_point(int(direction_profile_xyz))=(double(i)/double(number_k_points))*max_k_point;
		coulomb=pull(k_point);
		file_coulomb_potential<<i<<" "<<k_point(direction_profile_xyz)<<" "<<coulomb<<endl;
		k_point(direction_profile_xyz)=0.0;
	}
	file_coulomb_potential.close();
};
double Coulomb_Potential::pull(arma::vec k_point){
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
			coulomb_potential = conversion_parameter /(modulus_k_point+radius);
	}else if(dimension_potential==2){
		arma::vec primitive_along(3);
		for(int i=0;i<3;i++)
			if(direction_cutting(i)==0){
				primitive_along=primitive_vectors.col(i);
			}
		arma::vec k_point_orthogonal(3,arma::fill::zeros);
		arma::vec k_point_along(3,arma::fill::zeros);
		arma::vec unity(3,arma::fill::ones);
		k_point_along=(primitive_along%k_point)/norm(primitive_along);
		k_point_orthogonal=k_point-k_point_along;
		double c1=norm(k_point_along)/norm(k_point_orthogonal);
		double c2=(norm(primitive_along)/2)*norm(k_point_orthogonal);
		double c3=(norm(primitive_along)/2)*norm(k_point_along);
		if (modulus_k_point < minimum_k_point_modulus)
			coulomb_potential=0;
		else{
			coulomb_potential = conversion_parameter/(pow(modulus_k_point, 2)+radius);
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
	arma::mat k_points_list;
	arma::mat g_points_list;
	int htb_basis_dimension;
	int spin_htb_basis_dimension;
	int number_wannier_centers;
	arma::field<arma::mat> wannier_centers;
	int number_valence_bands;
	int number_conduction_bands;
	Hamiltonian_TB *hamiltonian_tb;
	double volume_cell;
	arma::mat bravais_lattice{arma::mat(3,3)};
	int spinorial_calculation;
	arma::vec number_points_real_space_grid{arma::vec(3)};
	arma::vec number_points_real_space_grid_percell{arma::vec(3)};
	arma::vec number_unit_cells_supercell{arma::vec(3)};
	int number_points_real_space_grid_total;
	arma::vec number_primitive_cells_integration{arma::vec(3)};
	arma::mat indexingi;
	arma::mat indexingj;
	arma::mat indexingk;
	arma::mat supercell_axis;
	arma::vec origin{arma::vec(3)};
	arma::vec origin_unitcell{arma::vec(3)};
	arma::mat real_space_wannier_functions_list;
public:
	Dipole_Elements(int number_k_points_list_tmp,arma::mat k_points_list_tmp, int number_g_points_list_tmp,arma::mat g_points_list_tmp,int number_wannier_centers_tmp,int number_valence_bands_selected_tmp,int number_conduction_bands_selected_tmp, Hamiltonian_TB *hamiltonian_tb_tmp,int spinorial_calculation_tmp,	Real_space_wannier* real_space_wannier_tmp,arma::vec number_primitive_cells_integration_tmp, arma::vec number_unit_cells_supercell_tmp, arma::vec number_points_real_space_grid_tmp);
	///arma::cx_mat function_building_exponential_factor(arma::vec excitonic_momentum_1,int minus,arma::vec excitonic_momentum_2);
	///arma::field<arma::mat> function_building_A_matrix(double threshold_proximity);
	arma::cx_vec function_building_real_space_wannier_dipole_ij(int number_wannier_1,int number_wannier_2,arma::vec excitonic_momentum,arma::vec g_momentum);
	arma::field<arma::cx_mat> function_building_M_k1k2_ij(arma::vec excitonic_momentum,int diagonal_k);
	///rho_{n1,n2,k1-p,k2-q}(excitonic_momentum,G)=\bra{n1k1-p}e^{i(excitonic_momentum+G)r\ket{n2k2-q}
	std::tuple<arma::cx_mat,arma::cx_mat,arma::cx_mat> pull_values(arma::vec excitonic_momentum,arma::vec parameter_l,arma::vec parameter_r,int diagonal_k,int minus,int left,int right,int reverse,int reverse_kk,double threshold_proximity);
////arma::mat function_translate(arma::mat wannier,int i,int j,int k);
	void print(arma::vec excitonic_momentum,arma::vec parameter_l,arma::vec parameter_r,int diagonal_k,int minus,int left, int right,double threshold_proximity);
	arma::mat pull_bravais_lattice(){
		return bravais_lattice;
	};
};
///arma::field<arma::mat> Dipole_Elements::function_building_A_matrix(double threshold_proximity){
///	arma::field<arma::mat> A_matrix(1);
///	(A_matrix(0)).set_size((spinorial_calculation+1)*number_wannier_centers,(spinorial_calculation+1)*number_wannier_centers);
///
///	///A is upper-diagonal (to avoid double-counting)
///	for(int spin=0;spin<spinorial_calculation+1;spin++)
///		for(int i=0;i<number_wannier_centers;i++)
///			for(int j=i;j<number_wannier_centers;j++)
///				if(arma::vecnorm(wannier_centers(spin).col(i)-wannier_centers(spin).col(j))<=threshold_proximity){
///					A_matrix(0)(spin*number_wannier_centers+i,spin*number_wannier_centers+j)=1.0;
///				}
///
///	return A_matrix;
///};
////// funzione di wannier scritta nel formalismo Wannier90
////// bisogna traslarla di un vettore del retticolo diretto (ijk)
///arma::mat Dipole_Elements:: function_translate(arma::mat wannier_in,int i,int j,int k){
///	arma::mat wannier_out((spinorial_calculation+1)*number_points_real_space_grid(0)*number_points_real_space_grid(1)*number_points_real_space_grid(2)/(number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)),number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2),arma::fill::zeros);
///	
///	////i2 --- > i2 - number_unit_cells_supercell(0)/2 --->  i2 - number_unit_cells_supercell(0)/2 < i*number_unit_cells_supercell(0)
///
///	for(int spin=0;spin<(spinorial_calculation+1);spin++)
///		for(int i2=0;i2<number_unit_cells_supercell(0);i2++)
///			for(int j2=0;j2<number_unit_cells_supercell(1);j2++)
///				for(int k2=0;k2<number_unit_cells_supercell(2);k2++){
///					if((i2<i)&&(j2<j)&&(k2<k))
///						for(int s2=0;s2<number_points_real_space_grid_percell(0);s2++)
///							for(int t2=0;t2<number_points_real_space_grid_percell(1);t2++)
///								for(int l2=0;l2<number_points_real_space_grid_percell(2);l2++)
///									wannier_out(spin*number_points_real_space_grid_total+s2*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t2*number_points_real_space_grid_percell(2)+l2,i2*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j2*number_unit_cells_supercell(2)+k2)=wannier_in(spin*number_points_real_space_grid_total+s2*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t2*number_points_real_space_grid_percell(2)+l2,(i2+i)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+(j2+j)*number_unit_cells_supercell(2)+(k2+k));
///				}
///	
///	return wannier_out;
///};


Dipole_Elements::Dipole_Elements(int number_k_points_list_tmp,arma::mat k_points_list_tmp, int number_g_points_list_tmp,arma::mat g_points_list_tmp, int number_wannier_centers_tmp, int number_valence_bands_tmp, int number_conduction_bands_tmp, Hamiltonian_TB *hamiltonian_tb_tmp,int spinorial_calculation_tmp, Real_space_wannier* real_space_wannier_tmp, arma::vec number_primitive_cells_integration_tmp, arma::vec number_unit_cells_supercell_tmp, arma::vec number_points_real_space_grid_tmp):
k_points_list(3,number_k_points_list_tmp), g_points_list(3,number_g_points_list_tmp), wannier_centers(spinorial_calculation_tmp+1), indexingi(number_primitive_cells_integration_tmp(0),number_unit_cells_supercell_tmp(0)), indexingj(number_primitive_cells_integration_tmp(1),number_unit_cells_supercell_tmp(1)), indexingk(number_primitive_cells_integration_tmp(2),number_unit_cells_supercell_tmp(2)), real_space_wannier_functions_list(number_points_real_space_grid_tmp(0)*number_points_real_space_grid_tmp(1)*number_points_real_space_grid_tmp(2)/(number_unit_cells_supercell_tmp(0)*number_unit_cells_supercell_tmp(1)*number_unit_cells_supercell_tmp(2)),number_unit_cells_supercell_tmp(0)*number_unit_cells_supercell_tmp(1)*number_unit_cells_supercell_tmp(2)*(spinorial_calculation_tmp+1)*number_wannier_centers_tmp)
{

	number_g_points_list=number_g_points_list_tmp;
	number_conduction_bands=number_conduction_bands_tmp;
	number_valence_bands=number_valence_bands_tmp;
	number_k_points_list=number_k_points_list_tmp;
	number_wannier_centers=number_wannier_centers_tmp;
	k_points_list=k_points_list_tmp;
	g_points_list=g_points_list_tmp;
	hamiltonian_tb=hamiltonian_tb_tmp;
	spinorial_calculation=spinorial_calculation_tmp;
	htb_basis_dimension=hamiltonian_tb->pull_htb_basis_dimension();
	bravais_lattice=hamiltonian_tb->pull_bravais_lattice();
	volume_cell=arma::det(hamiltonian_tb->pull_bravais_lattice());
	spin_htb_basis_dimension=htb_basis_dimension/(spinorial_calculation+1);
	
	number_points_real_space_grid=real_space_wannier_tmp->pull_number_points_real_space_grid();
	number_unit_cells_supercell=real_space_wannier_tmp->pull_number_unit_cells_supercell();
	origin=real_space_wannier_tmp->pull_origin();
	origin_unitcell=real_space_wannier_tmp->pull_origin_unitcell();
	supercell_axis=real_space_wannier_tmp->pull_supercell_axis();
	real_space_wannier_functions_list=real_space_wannier_tmp->pull_real_space_wannier_functions_list();

	for(int r=0;r<3;r++)
		number_points_real_space_grid_percell(r)=number_points_real_space_grid(r)/number_unit_cells_supercell(r);
	
	////number of primitive cells where to integrate
	number_primitive_cells_integration=number_primitive_cells_integration_tmp;
	number_points_real_space_grid_total=number_points_real_space_grid(0)*number_points_real_space_grid(1)*number_points_real_space_grid(2);

	if (spinorial_calculation == 1){
		//wannier_centers.set_size(2);
		wannier_centers(0).set_size(3,number_wannier_centers);
		wannier_centers(1).set_size(3,number_wannier_centers);
	}else{
		//wannier_centers.set_size(1);
		wannier_centers(0).set_size(3,number_wannier_centers);
	}
	wannier_centers=hamiltonian_tb->pull_wannier_centers();

	for(int i1=0;i1<number_primitive_cells_integration(0);i1++)
		for(int j1=0;j1<number_primitive_cells_integration(1);j1++)
			for(int k1=0;k1<number_primitive_cells_integration(2);k1++)
				for(int i2=0;i2<number_unit_cells_supercell(0);i2++)
					for(int j2=0;j2<number_unit_cells_supercell(1);j2++)
						for(int k2=0;k2<number_unit_cells_supercell(2);k2++){
							indexingi(i1,i2)=i2-(i1-int(number_primitive_cells_integration(0)/2));
							indexingj(j1,j2)=j2-(j1-int(number_primitive_cells_integration(1)/2));
							indexingk(k1,k2)=k2-(k1-int(number_primitive_cells_integration(2)/2));
							if(indexingi(i1,i2)>=number_unit_cells_supercell(0))
								indexingi(i1,i2)=-1;
							if(indexingj(j1,j2)>=number_unit_cells_supercell(1))
								indexingj(j1,j2)=-1;
							if(indexingk(k1,k2)>=number_unit_cells_supercell(2))
								indexingk(k1,k2)=-1;
						}

};
///arma::cx_mat Dipole_Elements::function_building_exponential_factor(arma::vec excitonic_momentum_1,int minus,arma::vec excitonic_momentum_2){
///	arma::cx_mat exponential_factor_tmp(htb_basis_dimension,number_g_points_list);
///	double temporary_variable;
///	arma::vec wannier_center(3);
///	for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
///		for(int g=0; g<number_g_points_list; g++)
///			for(int i=0; i<spin_htb_basis_dimension; i++){
///				temporary_variable=0.0;
///				wannier_center=(wannier_centers(spin_channel)).col(i);
///				for(int r=0; r<3; r++)
///					temporary_variable+=(wannier_center(r)*((1-minus*2)*(g_points_list(r,g)+excitonic_momentum_1(r)-excitonic_momentum_2(r))));
///				exponential_factor_tmp(spin_channel*spin_htb_basis_dimension+i,g).real(cos(temporary_variable));
///				exponential_factor_tmp(spin_channel*spin_htb_basis_dimension+i,g).imag(sin(temporary_variable));
///			}
///	////exponential_factor_tmp(spin_channel*spin_htb_basis_dimension+i,g).real(cos(accu((wannier_centers(spin_channel)).col(i)%((1-minus*2)*(g_points_list.col(g)+excitonic_momentum_1-excitonic_momentum_2)))));
///	////exponential_factor_tmp(spin_channel*spin_htb_basis_dimension+i,g).imag(sin(accu((wannier_centers(spin_channel)).col(i)%((1-minus*2)*(g_points_list.col(g)+excitonic_momentum_1-excitonic_momentum_2)))));
///	return exponential_factor_tmp;
///};

arma::cx_vec Dipole_Elements::function_building_real_space_wannier_dipole_ij(int number_wannier_1,int number_wannier_2,arma::vec excitonic_momentum,arma::vec g_momentum){
	arma::cx_vec A_matrix((spinorial_calculation+1)*number_primitive_cells_integration(0)*number_primitive_cells_integration(1)*number_primitive_cells_integration(2));
	arma::cx_double dipole; double exponent;
	double factor_conversion=1/(number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)*volume_cell);
	////\int dr w_1\sigma(r-R)e^{i(q+G)r}W_1\sigma(r)
	////it is necessary to translate the wannier function
	/// the strategy is to traslate it and putting to zero all elements outside the wannier supercell
	///wannier3=function_translate(wannier1,i1,j1,k1);
	////this is the integral over the wannier supercell
	int imax1=number_primitive_cells_integration(0);
	int jmax1=number_primitive_cells_integration(1);
	int kmax1=number_primitive_cells_integration(2);

	int count;
	///#pragma omp parallel for collapse(4) private(dipole,exponent) shared(A_matrix)
	for(int i1=0;i1<imax1;i1++)
		for(int j1=0;j1<jmax1;j1++)
			for(int k1=0;k1<kmax1;k1++)
				for(int spin=0;spin<(spinorial_calculation+1);spin++){
					dipole.real(0.0); dipole.imag(0.0);
					for(int i2=0;i2<number_unit_cells_supercell(0);i2++)
						for(int j2=0;j2<number_unit_cells_supercell(1);j2++)
							for(int k2=0;k2<number_unit_cells_supercell(2);k2++)
								if(indexingi(i1,i2)>=0&&indexingj(j1,j2)>=0&&indexingk(k1,k2)>=0)
								{
									for(int s2=0;s2<number_points_real_space_grid_percell(0);s2++)
										for(int t2=0;t2<number_points_real_space_grid_percell(1);t2++)
											for(int l2=0;l2<number_points_real_space_grid_percell(2);l2++){
												exponent=0;
												for(int r=0;r<3;r++)
													exponent+=(g_momentum(r)+excitonic_momentum(r))*(origin(r)+(i2+s2/number_points_real_space_grid_percell(0))*supercell_axis(r,0)+(j2+t2/number_points_real_space_grid_percell(1))*supercell_axis(r,1)+(k2+l2/number_points_real_space_grid_percell(2))*supercell_axis(r,2));
												dipole.real(dipole.real()+cos(exponent)*(real_space_wannier_functions_list(s2*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t2*number_points_real_space_grid_percell(2)+l2,spin*number_wannier_centers*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+number_wannier_1*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+indexingi(i1,i2)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+indexingj(j1,j2)*number_unit_cells_supercell(2)+indexingk(k1,k2)))*(real_space_wannier_functions_list(s2*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t2*number_points_real_space_grid_percell(2)+l2,spin*number_wannier_centers*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+number_wannier_2*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i2*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j2*number_unit_cells_supercell(2)+k2)));			
												dipole.imag(dipole.imag()+sin(exponent)*(real_space_wannier_functions_list(s2*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t2*number_points_real_space_grid_percell(2)+l2,spin*number_wannier_centers*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+number_wannier_1*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+indexingi(i1,i2)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+indexingj(j1,j2)*number_unit_cells_supercell(2)+indexingk(k1,k2)))*(real_space_wannier_functions_list(s2*number_points_real_space_grid_percell(1)*number_points_real_space_grid_percell(2)+t2*number_points_real_space_grid_percell(2)+l2,spin*number_wannier_centers*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+number_wannier_2*number_unit_cells_supercell(0)*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+i2*number_unit_cells_supercell(1)*number_unit_cells_supercell(2)+j2*number_unit_cells_supercell(2)+k2)));			
											}
								}
					A_matrix(spin*number_primitive_cells_integration(0)*number_primitive_cells_integration(1)*number_primitive_cells_integration(2)+i1*number_primitive_cells_integration(1)*number_primitive_cells_integration(2)+j1*number_primitive_cells_integration(2)+k1)=factor_conversion*dipole/double(number_points_real_space_grid_total);
				}	
	return A_matrix;
};

arma::field<arma::cx_mat> Dipole_Elements::function_building_M_k1k2_ij(arma::vec excitonic_momentum,int diagonal_k){
	cout<<"starting M"<<endl;

	arma::cx_vec A_matrix1((spinorial_calculation+1)*number_primitive_cells_integration(0)*number_primitive_cells_integration(1)*number_primitive_cells_integration(2));
	///	arma::cx_mat A_matrix2((spinorial_calculation+1)*number_primitive_cells_integration(0)*number_primitive_cells_integration(1)*number_primitive_cells_integration(2),number_k_points_list*number_k_points_list*number_g_points_list*number_wannier_centers*number_wannier_centers);
	arma::cx_double composition;
	arma::cx_double temporary;
	double exponent;
	
	if(diagonal_k==1){
		arma::field<arma::cx_mat> M_matrix(number_g_points_list);
		for(int i=0;i<number_g_points_list;i++)
				M_matrix(i).set_size((spinorial_calculation+1)*number_wannier_centers,(spinorial_calculation+1)*number_wannier_centers);
		
		#pragma omp parallel for collapse(3) private(A_matrix1,composition,exponent,temporary) shared(M_matrix)
		for(int w1=0;w1<number_wannier_centers;w1++)
			for(int w2=0;w2<number_wannier_centers;w2++)
				for(int i=0;i<number_g_points_list;i++){
					A_matrix1=function_building_real_space_wannier_dipole_ij(w1,w2,excitonic_momentum,g_points_list.col(i));
					for(int spin=0;spin<(spinorial_calculation+1);spin++){
						composition.real(0.0); composition.imag(0.0);
						for(int l1=0;l1<number_primitive_cells_integration(0);l1++)
							for(int j1=0;j1<number_primitive_cells_integration(1);j1++)
								for(int k1=0;k1<number_primitive_cells_integration(2);k1++){
									exponent=0;
									for(int r=0;r<3;r++)
										exponent+=((excitonic_momentum(r))*((l1-int(number_primitive_cells_integration(0)/2))*bravais_lattice(r,0)+(j1-int(number_primitive_cells_integration(1)/2))*bravais_lattice(r,1)+(k1-int(number_primitive_cells_integration(2)/2))*bravais_lattice(r,2)));
										//exponent=arma::accu((excitonic_momentum.t())*((l1-int(number_primitive_cells_integration(0)/2))*bravais_lattice.col(0)+(j1-int(number_primitive_cells_integration(1)/2))*bravais_lattice.col(1)+(k1-int(number_primitive_cells_integration(2)/2))*bravais_lattice.col(2)));
									temporary=A_matrix1(spin*number_primitive_cells_integration(0)*number_primitive_cells_integration(1)*number_primitive_cells_integration(2)+l1*number_primitive_cells_integration(1)*number_primitive_cells_integration(2)+j1*number_primitive_cells_integration(2)+k1);
									composition.real(composition.real()+cos(exponent)*real(temporary)-sin(exponent)*imag(temporary));
									composition.imag(composition.imag()-sin(exponent)*real(temporary)+cos(exponent)*imag(temporary));
								}
						M_matrix(i)(spin*number_wannier_centers+w1,spin*number_wannier_centers+w2)=composition;
						////(number_primitive_cells_integration(0)*number_primitive_cells_integration(1)*number_primitive_cells_integration(2));
					}
			}
		#pragma omp parallel for collapse(4) 
		for(int w1=0;w1<number_wannier_centers;w1++)
			for(int w2=w1;w2<number_wannier_centers;w2++)
				for(int i=0;i<number_g_points_list;i++)
					for(int spin=0;spin<(spinorial_calculation+1);spin++)
						M_matrix(i)(spin*number_wannier_centers+w2,spin*number_wannier_centers+w1)=conj(M_matrix(i)(spin*number_wannier_centers+w1,spin*number_wannier_centers+w2));
		cout<<"finishing M"<<endl;
		return M_matrix;

	}else{
		arma::field<arma::cx_mat> M_matrix(number_g_points_list*number_k_points_list*number_k_points_list);
		for(int i=0;i<number_g_points_list;i++)
			for(int k1=0;k1<number_k_points_list;k1++)
				for(int k2=0;k2<number_k_points_list;k2++)
					M_matrix(i*number_k_points_list*number_k_points_list+k1*number_k_points_list+k2).set_size((spinorial_calculation+1)*number_wannier_centers,(spinorial_calculation+1)*number_wannier_centers);
		double norm=arma::vecnorm(excitonic_momentum);
		if((norm==0)&&(number_g_points_list==1)){
			cout<<"faster approach"<<endl;
			#pragma omp parallel for collapse(5) private(A_matrix1,composition,exponent,temporary) shared(M_matrix)
			for(int i=0;i<number_g_points_list;i++)	
				for(int w1=0;w1<number_wannier_centers;w1++)
					for(int w2=w1;w2<number_wannier_centers;w2++)	
						for(int k1=0;k1<number_k_points_list;k1++)
							for(int k2=k1;k2<number_k_points_list;k2++){
								///cout<<w1<<" "<<w2<<" "<<k1<<" "<<k2<<endl;
								//A_matrix2.col(k1*number_k_points_list*number_g_points_list*number_wannier_centers*number_wannier_centers+k2*number_g_points_list*number_wannier_centers*number_wannier_centers+i*number_wannier_centers*number_wannier_centers+w1*number_wannier_centers+w2)=function_building_real_space_wannier_dipole_ij(w1,w2,excitonic_momentum+k_points_list.col(k1)-k_points_list.col(k2),g_points_list.col(i));
								A_matrix1=function_building_real_space_wannier_dipole_ij(w1,w2,excitonic_momentum+k_points_list.col(k1)-k_points_list.col(k2),g_points_list.col(i));		
								composition.real(0.0); 
								composition.imag(0.0);
								for(int spin=0;spin<(spinorial_calculation+1);spin++){
									for(int l1=0;l1<number_primitive_cells_integration(0);l1++)
										for(int j1=0;j1<number_primitive_cells_integration(1);j1++)
											for(int k1=0;k1<number_primitive_cells_integration(2);k1++){
												exponent=arma::accu(((excitonic_momentum+k_points_list.col(k1)-k_points_list.col(k2)).t())*(l1*bravais_lattice.col(0)+j1*bravais_lattice.col(1)+k1*bravais_lattice.col(2)));
												temporary=A_matrix1(spin*number_primitive_cells_integration(0)*number_primitive_cells_integration(1)*number_primitive_cells_integration(2)+l1*number_primitive_cells_integration(1)*number_primitive_cells_integration(2)+j1*number_primitive_cells_integration(2)+k1);
												composition.real(composition.real()+cos(exponent)*real(temporary)-sin(exponent)*imag(temporary));
												composition.imag(composition.imag()-sin(exponent)*real(temporary)+cos(exponent)*imag(temporary));
											}
									M_matrix(i*number_k_points_list+k1*number_k_points_list+k2)(spin*number_wannier_centers+w1,spin*number_wannier_centers+w2)=composition/(number_primitive_cells_integration(0)*number_primitive_cells_integration(1)*number_primitive_cells_integration(2));
								}
							}
				
			#pragma omp parallel for collapse(6) shared(M_matrix)			
			for(int i=0;i<number_g_points_list;i++)	
				for(int w1=0;w1<number_wannier_centers-1;w1++)
					for(int w2=w1+1;w2<number_wannier_centers;w2++)	
						for(int k1=0;k1<number_k_points_list-1;k1++)
							for(int k2=k1+1;k2<number_k_points_list;k2++)
								for(int spin=0;spin<(spinorial_calculation+1);spin++)
									M_matrix(i*number_k_points_list+k2*number_k_points_list+k1)(spin*number_wannier_centers+w2,spin*number_wannier_centers+w1)=conj(M_matrix(i*number_k_points_list+k1*number_k_points_list+k2)(spin*number_wannier_centers+w1,spin*number_wannier_centers+w2));
		}else{
			#pragma omp parallel for collapse(5) private(A_matrix1,composition,exponent,temporary) shared(M_matrix)
			for(int i=0;i<number_g_points_list;i++)	
				for(int w1=0;w1<number_wannier_centers;w1++)
					for(int w2=0;w2<number_wannier_centers;w2++)	
						for(int k1=0;k1<number_k_points_list;k1++)
							for(int k2=0;k2<number_k_points_list;k2++){
								///cout<<w1<<" "<<w2<<" "<<k1<<" "<<k2<<endl;
								//A_matrix2.col(k1*number_k_points_list*number_g_points_list*number_wannier_centers*number_wannier_centers+k2*number_g_points_list*number_wannier_centers*number_wannier_centers+i*number_wannier_centers*number_wannier_centers+w1*number_wannier_centers+w2)=function_building_real_space_wannier_dipole_ij(w1,w2,excitonic_momentum+k_points_list.col(k1)-k_points_list.col(k2),g_points_list.col(i));
								A_matrix1=function_building_real_space_wannier_dipole_ij(w1,w2,excitonic_momentum+k_points_list.col(k1)-k_points_list.col(k2),g_points_list.col(i));		
								composition.real(0.0); 
								composition.imag(0.0);
								for(int spin=0;spin<(spinorial_calculation+1);spin++){
									for(int l1=0;l1<number_primitive_cells_integration(0);l1++)
										for(int j1=0;j1<number_primitive_cells_integration(1);j1++)
											for(int k1=0;k1<number_primitive_cells_integration(2);k1++){
												exponent=arma::accu(((excitonic_momentum+k_points_list.col(k1)-k_points_list.col(k2)).t())*(l1*bravais_lattice.col(0)+j1*bravais_lattice.col(1)+k1*bravais_lattice.col(2)));
												temporary=A_matrix1(spin*number_primitive_cells_integration(0)*number_primitive_cells_integration(1)*number_primitive_cells_integration(2)+l1*number_primitive_cells_integration(1)*number_primitive_cells_integration(2)+j1*number_primitive_cells_integration(2)+k1);
												composition.real(composition.real()+cos(exponent)*real(temporary)-sin(exponent)*imag(temporary));
												composition.imag(composition.imag()-sin(exponent)*real(temporary)+cos(exponent)*imag(temporary));
											}
									M_matrix(i*number_k_points_list+k1*number_k_points_list+k2)(spin*number_wannier_centers+w1,spin*number_wannier_centers+w2)=composition/(number_primitive_cells_integration(0)*number_primitive_cells_integration(1)*number_primitive_cells_integration(2));
								}
							}
		}
		cout<<"finishing M"<<endl;
		return M_matrix;
	}
};

///diagonal_k ---> rho_{n1,n2,k1-p,k2-q}(excitonic_momentum,G)-->rho_{n1,n2,k1-p,k1-q}(excitonic_momentum,G)
std::tuple<arma::cx_mat,arma::cx_mat,arma::cx_mat> Dipole_Elements::pull_values(arma::vec excitonic_momentum,arma::vec parameter_l,arma::vec parameter_r,int diagonal_k,int minus,int left,int right,int reverse, int reverse_kk,double threshold_proximity){
	arma::vec zeros_vec(3); int effective_number_k_points_list;
	///adding exponential term e^{i(k+G)r} to the right states
	///e_{gl}k_{gm} -> l_{g(l,m)}
	///calculating the exponenential factor at the beginning
	
	if(diagonal_k==1)
		effective_number_k_points_list=number_k_points_list;
	else
		effective_number_k_points_list=number_k_points_list*number_k_points_list;
	
	int number_left_states=left*number_conduction_bands+(1-left)*number_valence_bands;
	int number_right_states=right*number_conduction_bands+(1-right)*number_valence_bands;
	arma::cx_mat energies_diff((spinorial_calculation+1),number_left_states*number_right_states*number_k_points_list);
	arma::cx_mat energies_sum((spinorial_calculation+1),number_left_states*number_right_states*number_k_points_list);
	arma::cx_mat rho((spinorial_calculation+1)*number_left_states*number_right_states*effective_number_k_points_list,number_g_points_list,arma::fill::zeros);
	

	///this is the method using proximity of the wannier functions, and putting to 1 every dipole elements in the same site
	///arma::field<arma::mat> A_matrix=function_building_A_matrix(threshold_proximity);
	///dimension a little bit different
	arma::field<arma::cx_mat> A_matrix=function_building_M_k1k2_ij(excitonic_momentum,diagonal_k);
///	cout<<A_matrix<<endl;
	///in order to avoid twice the calculations in the case of diagonal_k=0, the two possibilities have been separated
	////this is the heaviest but also the fastest solution (to use cx_cub for left state instead of cx_mat)
	if(diagonal_k==1){
		///arma::cx_mat exponential_factor=function_building_exponential_factor(excitonic_momentum,minus,zeros_vec);
		////initializing memory
		arma::cx_mat ks_state_l(htb_basis_dimension, number_left_states); 
		arma::cx_mat ks_state_r(htb_basis_dimension, number_right_states); 
		arma::mat ks_energy_l(2,number_left_states);
		arma::mat ks_energy_r(2,number_right_states);
		std::tuple<arma::mat,arma::cx_mat> ks_state_l_k_points(ks_energy_l,ks_state_l); 
		std::tuple<arma::mat,arma::cx_mat> ks_state_r_k_points(ks_energy_r,ks_state_r);
		arma::cx_cube ks_state_right(htb_basis_dimension,number_right_states*effective_number_k_points_list,number_g_points_list,arma::fill::zeros);
		arma::cx_cube ks_state_left(htb_basis_dimension,number_left_states*effective_number_k_points_list,number_g_points_list,arma::fill::zeros);

		///private(ks_states_k_point,ks_states_k_point_q,ks_state,ks_state_q,ks_energy,ks_energy_q)
		for(int i=0;i<number_k_points_list;i++){
			ks_state_l_k_points = hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i)-parameter_l,(1-left)*number_valence_bands,left*number_conduction_bands);
			ks_state_r_k_points = hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i)-parameter_r,(1-right)*number_valence_bands,right*number_conduction_bands);	
			ks_state_l=get<1>(ks_state_l_k_points); 
			ks_state_r=get<1>(ks_state_r_k_points);
			ks_energy_l=get<0>(ks_state_l_k_points); 
			ks_energy_r=get<0>(ks_state_r_k_points);
			for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++){
				for(int g=0;g<number_g_points_list;g++){
					for(int m=0;m<number_right_states;m++)
						for(int r=0;r<spin_htb_basis_dimension;r++){
							///ks_state_right(spin_channel*spin_htb_basis_dimension+r,m*number_k_points_list+i,g).real(real(exponential_factor(spin_channel*spin_htb_basis_dimension+r,g)*ks_state_r(spin_channel*spin_htb_basis_dimension+r,m)));
							///ks_state_right(spin_channel*spin_htb_basis_dimension+r,m*number_k_points_list+i,g).imag(imag(exponential_factor(spin_channel*spin_htb_basis_dimension+r,g)*ks_state_r(spin_channel*spin_htb_basis_dimension+r,m)));
							ks_state_right(spin_channel*spin_htb_basis_dimension+r,m*number_k_points_list+i,g).real(real(ks_state_r(spin_channel*spin_htb_basis_dimension+r,m)));
							ks_state_right(spin_channel*spin_htb_basis_dimension+r,m*number_k_points_list+i,g).imag(imag(ks_state_r(spin_channel*spin_htb_basis_dimension+r,m)));
						}
					for(int n=0;n<number_left_states;n++)	
						ks_state_left.subcube(spin_channel*spin_htb_basis_dimension,n*number_k_points_list+i,g,(spin_channel+1)*spin_htb_basis_dimension-1,n*number_k_points_list+i,g)=
							ks_state_l.submat(spin_channel*spin_htb_basis_dimension,n,(spin_channel+1)*spin_htb_basis_dimension-1,n);
					}
				for(int n=0;n<number_left_states;n++)
					for(int m=0;m<number_right_states;m++){
						energies_diff(spin_channel,n*number_right_states*number_k_points_list+m*number_k_points_list+i).real(ks_energy_l(spin_channel,n)-ks_energy_r(spin_channel,m));
						energies_sum(spin_channel,n*number_right_states*number_k_points_list+m*number_k_points_list+i).real(ks_energy_l(spin_channel,n)+ks_energy_r(spin_channel,m));
					}
			}
		}
		//cout<<ks_state_left<<endl;
		//cout<<ks_state_right<<endl;
		arma::cx_double normalization;
		arma::cx_double temporary;
		///if(number_g_points_list==1){
		if(reverse==0){
			//#pragma omp parallel for collapse(6) 
			for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
				for(int n=0;n<number_left_states;n++)
					for(int m=0;m<number_right_states;m++)
						for(int g=0;g<number_g_points_list;g++)
							for(int i=0;i<number_k_points_list;i++){
								normalization.real(0.0); normalization.imag(0.0);
								temporary.real(0.0); temporary.imag(0.0);
								for(int r=0;r<spin_htb_basis_dimension;r++){
									////normalization+=conj(ks_state_left(spin_channel*spin_htb_basis_dimension+r,n*number_k_points_list+i,g))*ks_state_right(spin_channel*spin_htb_basis_dimension+r,m*number_k_points_list+i,g);
									for(int s=0;s<spin_htb_basis_dimension;s++)
										temporary+=conj(ks_state_left(spin_channel*spin_htb_basis_dimension+r,n*number_k_points_list+i,g))*A_matrix(g)(spin_channel*number_wannier_centers+r,spin_channel*number_wannier_centers+s)*ks_state_right(spin_channel*spin_htb_basis_dimension+s,m*number_k_points_list+i,g);
											///rho(spin_channel*number_left_states*number_right_states*number_k_points_list+n*number_right_states*number_k_points_list+m*number_k_points_list+i,g).real((rho(spin_channel*number_left_states*number_right_states*number_k_points_list+n*number_right_states*number_k_points_list+m*number_k_points_list+i,g)).real()+real(conj(ks_state_left(spin_channel*spin_htb_basis_dimension+r,n*number_k_points_list+i,g))*A_matrix(g)(spin_channel*number_wannier_centers+r,spin_channel*number_wannier_centers+s)*ks_state_right(spin_channel*spin_htb_basis_dimension+s,m*number_k_points_list+i,g)));
										///rho(spin_channel*number_left_states*number_right_states*number_k_points_list+n*number_right_states*number_k_points_list+m*number_k_points_list+i,g).imag((rho(spin_channel*number_left_states*number_right_states*number_k_points_list+n*number_right_states*number_k_points_list+m*number_k_points_list+i,g)).imag()+imag(conj(ks_state_left(spin_channel*spin_htb_basis_dimension+r,n*number_k_points_list+i,g))*A_matrix(g)(spin_channel*number_wannier_centers+r,spin_channel*number_wannier_centers+s)*ks_state_right(spin_channel*spin_htb_basis_dimension+s,m*number_k_points_list+i,g)));
								rho(spin_channel*number_left_states*number_right_states*number_k_points_list+n*number_right_states*number_k_points_list+m*number_k_points_list+i,g)=temporary;
								}
							}
		}else{
			///#pragma omp parallel for collapse(6) 
			for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
				for(int n=0;n<number_left_states;n++)
					for(int m=0;m<number_right_states;m++)
						for(int g=0;g<number_g_points_list;g++)
							for(int i=0;i<number_k_points_list;i++)
								for(int r=0;r<spin_htb_basis_dimension;r++)
									for(int s=0;s<spin_htb_basis_dimension;s++){
										rho(spin_channel*number_left_states*number_right_states*number_k_points_list+m*number_left_states*number_k_points_list+n*number_k_points_list+i,g).real((rho(spin_channel*number_left_states*number_right_states*number_k_points_list+m*number_left_states*number_k_points_list+n*number_k_points_list+i,g)).real()+real(conj(ks_state_left(spin_channel*spin_htb_basis_dimension+r,n*number_k_points_list+i,g))*A_matrix(g)(spin_channel*number_wannier_centers+r,spin_channel*number_wannier_centers+s)*ks_state_right(spin_channel*spin_htb_basis_dimension+s,m*number_k_points_list+i,g)));
										rho(spin_channel*number_left_states*number_right_states*number_k_points_list+m*number_left_states*number_k_points_list+n*number_k_points_list+i,g).imag((rho(spin_channel*number_left_states*number_right_states*number_k_points_list+m*number_left_states*number_k_points_list+n*number_k_points_list+i,g)).imag()+imag(conj(ks_state_left(spin_channel*spin_htb_basis_dimension+r,n*number_k_points_list+i,g))*A_matrix(g)(spin_channel*number_wannier_centers+r,spin_channel*number_wannier_centers+s)*ks_state_right(spin_channel*spin_htb_basis_dimension+s,m*number_k_points_list+i,g)));
								}
		}
		///}else{
		///	if(reverse==0){
		///		for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
		///			for(int n=0;n<number_left_states;n++)
		///				for(int m=0;m<number_right_states;m++)
		///					rho.submat(spin_channel*number_left_states*number_right_states*number_k_points_list+n*number_right_states*number_k_points_list+m*number_k_points_list,0,spin_channel*number_left_states*number_right_states*number_k_points_list+n*number_right_states*number_k_points_list+m*number_k_points_list+number_k_points_list-1,number_g_points_list-1)=
		///						((cx_mat)(sum(conj(ks_state_left.subcube(spin_channel*spin_htb_basis_dimension,n*number_k_points_list,0,(spin_channel+1)*spin_htb_basis_dimension-1,(n+1)*number_k_points_list-1,number_g_points_list-1))%
		///						ks_state_right.subcube(spin_channel*spin_htb_basis_dimension,m*number_k_points_list,0,(spin_channel+1)*spin_htb_basis_dimension-1,(m+1)*number_k_points_list-1,number_g_points_list-1),0)));
		///	}else{
		///		for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
		///			for(int n=0;n<number_left_states;n++)
		///				for(int m=0;m<number_right_states;m++)
		///					rho.submat(spin_channel*number_left_states*number_right_states*number_k_points_list+m*number_left_states*number_k_points_list+n*number_k_points_list,0,spin_channel*number_left_states*number_right_states*number_k_points_list+m*number_left_states*number_k_points_list+n*number_k_points_list+number_k_points_list-1,number_g_points_list-1)=
		///						((cx_mat)(sum(conj(ks_state_left.subcube(spin_channel*spin_htb_basis_dimension,n*number_k_points_list,0,(spin_channel+1)*spin_htb_basis_dimension-1,(n+1)*number_k_points_list-1,number_g_points_list-1))%
		///						ks_state_right.subcube(spin_channel*spin_htb_basis_dimension,m*number_k_points_list,0,(spin_channel+1)*spin_htb_basis_dimension-1,(m+1)*number_k_points_list-1,number_g_points_list-1),0)));
		///	}
		///}

		ks_state_r.reset();
		ks_state_l.reset();
		ks_state_right.reset();
		ks_state_left.reset();
	///	exponential_factor.reset();
		ks_energy_l.reset();
		ks_energy_r.reset();
	}else{
		arma::cx_mat energies((spinorial_calculation+1),effective_number_k_points_list*number_left_states*number_right_states,arma::fill::zeros);
		arma::cx_cube ks_state_l_k_points(htb_basis_dimension,number_left_states,number_k_points_list);
		arma::cx_cube ks_state_r_k_points(htb_basis_dimension,number_right_states,number_k_points_list);
		
		cout<<"diagonalization"<<endl;
		for(int i=0;i<number_k_points_list;i++){
			ks_state_l_k_points.slice(i) = get<1>(hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i)-parameter_l,(1-left)*number_valence_bands,left*number_conduction_bands));
			ks_state_r_k_points.slice(i) = get<1>(hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i)-parameter_r,(1-right)*number_valence_bands,right*number_conduction_bands));
		}
		///cout<<"exponential_factor"<<endl;
		//////arma::cx_cube exponential_factor(htb_basis_dimension,number_g_points_list,number_k_points_list*number_k_points_list);
		///arma::cx_mat temporary_value(htb_basis_dimension,number_g_points_list);
		///if(reverse_kk==0){
		///	//#pragma omp parallel for collapse(2)
		///	for(int i=0;i<number_k_points_list;i++)
		///		for(int j=0;j<number_k_points_list;j++){
		///			temporary_value=function_building_exponential_factor(k_points_list.col(i)-excitonic_momentum,minus,k_points_list.col(j));
		///			for(int h=0;h<htb_basis_dimension;h++)
		///				for(int r=0;r<number_g_points_list;r++)
		///					exponential_factor(h,r,i*number_k_points_list+j)=temporary_value(h,r);
		///		}
		///}else{
		///	///#pragma omp parallel for collapse(2)
		///	for(int i=0;i<number_k_points_list;i++)
		///		for(int j=0;j<number_k_points_list;j++){
		///			temporary_value=function_building_exponential_factor(k_points_list.col(j)-excitonic_momentum,minus,k_points_list.col(i));
		///			for(int h=0;h<htb_basis_dimension;h++)
		///				for(int r=0;r<number_g_points_list;r++)
		///					exponential_factor(h,r,i*number_k_points_list+j)=temporary_value(h,r);
		///	}
		///}	
			
		
		///cout<<exponential_factor<<endl;
		cout<<"combination"<<endl;
		if(reverse==0){
			//#pragma omp parallel for collapse(7)
			for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
				for(int n=0;n<number_left_states;n++)
					for(int m=0;m<number_right_states;m++)
						for(int i=0;i<number_k_points_list;i++)
							for(int j=0;j<number_k_points_list;j++)
								for(int g=0;g<number_g_points_list;g++)
									for(int r=0;r<spin_htb_basis_dimension;r++)
										for(int s=0;s<spin_htb_basis_dimension;s++){
											///rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+n*number_right_states*effective_number_k_points_list+m*effective_number_k_points_list+i*number_k_points_list+j,g).real((rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+n*number_right_states*effective_number_k_points_list+m*effective_number_k_points_list+i*number_k_points_list+j,g)).real()+real(exponential_factor(spin_channel*spin_htb_basis_dimension+r,g,i*number_k_points_list+j)*(ks_state_r_k_points(spin_channel*spin_htb_basis_dimension+r,m,j))*A_matrix(g*number_k_points_list*number_k_points_list+i*number_k_points_list+j)(r,s)*conj(ks_state_l_k_points(spin_channel*spin_htb_basis_dimension+s,n,i))));
											///rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+n*number_right_states*effective_number_k_points_list+m*effective_number_k_points_list+i*number_k_points_list+j,g).imag((rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+n*number_right_states*effective_number_k_points_list+m*effective_number_k_points_list+i*number_k_points_list+j,g)).imag()+imag(exponential_factor(spin_channel*spin_htb_basis_dimension+r,g,i*number_k_points_list+j)*(ks_state_r_k_points(spin_channel*spin_htb_basis_dimension+r,m,j))*A_matrix(g*number_k_points_list*number_k_points_list+i*number_k_points_list+j)(r,s)*conj(ks_state_l_k_points(spin_channel*spin_htb_basis_dimension+r,n,i))));
											rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+n*number_right_states*effective_number_k_points_list+m*effective_number_k_points_list+i*number_k_points_list+j,g).real((rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+n*number_right_states*effective_number_k_points_list+m*effective_number_k_points_list+i*number_k_points_list+j,g)).real()+real((ks_state_r_k_points(spin_channel*spin_htb_basis_dimension+r,m,j))*A_matrix(g*number_k_points_list*number_k_points_list+i*number_k_points_list+j)(spin_channel*number_wannier_centers+r,spin_channel*number_wannier_centers+s)*conj(ks_state_l_k_points(spin_channel*spin_htb_basis_dimension+s,n,i))));
											rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+n*number_right_states*effective_number_k_points_list+m*effective_number_k_points_list+i*number_k_points_list+j,g).imag((rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+n*number_right_states*effective_number_k_points_list+m*effective_number_k_points_list+i*number_k_points_list+j,g)).imag()+imag((ks_state_r_k_points(spin_channel*spin_htb_basis_dimension+r,m,j))*A_matrix(g*number_k_points_list*number_k_points_list+i*number_k_points_list+j)(spin_channel*number_wannier_centers+r,spin_channel*number_wannier_centers+s)*conj(ks_state_l_k_points(spin_channel*spin_htb_basis_dimension+r,n,i))));
									}		
		}else{
			//#pragma omp parallel for collapse(7) 
			for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
				for(int n=0;n<number_left_states;n++)
					for(int m=0;m<number_right_states;m++)
						for(int i=0;i<number_k_points_list;i++)
							for(int j=0;j<number_k_points_list;j++)
								for(int g=0;g<number_g_points_list;g++)
									for(int r=0;r<spin_htb_basis_dimension;r++)
										for(int s=0;s<spin_htb_basis_dimension;s++){
											///rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+m*number_left_states*effective_number_k_points_list+n*effective_number_k_points_list+j*number_k_points_list+i,g).real((rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+m*number_left_states*effective_number_k_points_list+n*effective_number_k_points_list+j*number_k_points_list+i,g)).real()+real(exponential_factor(spin_channel*spin_htb_basis_dimension+r,g,i*number_k_points_list+j)*(ks_state_r_k_points(spin_channel*spin_htb_basis_dimension+r,m,j))*A_matrix(g*number_k_points_list*number_k_points_list+i*number_k_points_list+j)(r,s)*conj(ks_state_l_k_points(spin_channel*spin_htb_basis_dimension+s,n,i))));
											///rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+m*number_left_states*effective_number_k_points_list+n*effective_number_k_points_list+j*number_k_points_list+i,g).imag((rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+m*number_left_states*effective_number_k_points_list+n*effective_number_k_points_list+j*number_k_points_list+i,g)).imag()+imag(exponential_factor(spin_channel*spin_htb_basis_dimension+r,g,i*number_k_points_list+j)*(ks_state_r_k_points(spin_channel*spin_htb_basis_dimension+r,m,j))*A_matrix(g*number_k_points_list*number_k_points_list+i*number_k_points_list+j)(r,s)*conj(ks_state_l_k_points(spin_channel*spin_htb_basis_dimension+s,n,i))));
											rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+m*number_left_states*effective_number_k_points_list+n*effective_number_k_points_list+j*number_k_points_list+i,g).real((rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+m*number_left_states*effective_number_k_points_list+n*effective_number_k_points_list+j*number_k_points_list+i,g)).real()+real((ks_state_r_k_points(spin_channel*spin_htb_basis_dimension+r,m,j))*A_matrix(g*number_k_points_list*number_k_points_list+i*number_k_points_list+j)(spin_channel*number_wannier_centers+r,spin_channel*number_wannier_centers+s)*conj(ks_state_l_k_points(spin_channel*spin_htb_basis_dimension+s,n,i))));
											rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+m*number_left_states*effective_number_k_points_list+n*effective_number_k_points_list+j*number_k_points_list+i,g).imag((rho(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+m*number_left_states*effective_number_k_points_list+n*effective_number_k_points_list+j*number_k_points_list+i,g)).imag()+imag((ks_state_r_k_points(spin_channel*spin_htb_basis_dimension+r,m,j))*A_matrix(g*number_k_points_list*number_k_points_list+i*number_k_points_list+j)(spin_channel*number_wannier_centers+r,spin_channel*number_wannier_centers+s)*conj(ks_state_l_k_points(spin_channel*spin_htb_basis_dimension+s,n,i))));
										}	
		}
		//for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++)
		//	for(int j=0;j<number_k_points_list;j++){
		//		ks_state=ks_state_r_k_points.slice(j);
		//		#pragma omp parallel for collapse(3)
		//		for(int n=0;n<number_left_states;n++)
		//			for(int m=0;m<number_right_states;m++)
		//				for(int i=0;i<number_k_points_list;i++){
		//					for(int g=0;g<number_g_points_list;g++){
		//						ks_state_right.subcube(spin_channel*spin_htb_basis_dimension,i*number_k_points_list+j,g,(spin_channel+1)*spin_htb_basis_dimension-1,i*number_k_points_list+j,g)=
		//							(cx_mat)(exponential_factor.subcube(spin_channel*spin_htb_basis_dimension,g,i*number_k_points_list+j,(spin_channel+1)*spin_htb_basis_dimension-1,g,i*number_k_points_list+j))%ks_state.submat(spin_channel*spin_htb_basis_dimension,m,(spin_channel+1)*spin_htb_basis_dimension-1,m);
		//						ks_state_left.subcube(spin_channel*spin_htb_basis_dimension,i*number_k_points_list+j,g,(spin_channel+1)*spin_htb_basis_dimension-1,i*number_k_points_list+j,g)=
		//							ks_state_l_k_points.subcube(spin_channel*spin_htb_basis_dimension,n,i,(spin_channel+1)*spin_htb_basis_dimension-1,n,i);
		//					}
		//					rho.submat(spin_channel*number_left_states*number_right_states*effective_number_k_points_list+m*number_left_states*effective_number_k_points_list+n*effective_number_k_points_list,0,spin_channel*number_left_states*number_right_states*effective_number_k_points_list+m*number_left_states*effective_number_k_points_list+n*effective_number_k_points_list+effective_number_k_points_list-1,number_g_points_list-1)=
		//						((cx_mat)(sum(conj(ks_state_left.subcube(spin_channel*spin_htb_basis_dimension,0,0,(spin_channel+1)*spin_htb_basis_dimension-1,effective_number_k_points_list-1,number_g_points_list-1))%
		//						ks_state_right.subcube(spin_channel*spin_htb_basis_dimension,0,0,(spin_channel+1)*spin_htb_basis_dimension-1,effective_number_k_points_list-1,number_g_points_list-1),0)));		
		//				}	
		//	}
		
		ks_state_r_k_points.reset();
		ks_state_l_k_points.reset();
	///	exponential_factor.reset();
	}
	return {energies_diff,energies_sum,rho};
};

void Dipole_Elements::print(arma::vec excitonic_momentum,arma::vec parameter_l,arma::vec parameter_r,int diagonal_k,int minus,int left, int right,double threshold_proximity){
	
	///std::tuple<arma::cx_mat,arma::cx_mat,arma::cx_mat> energies_and_dipole_elements=pull_values(excitonic_momentum,parameter_l,parameter_r,diagonal_k,minus,left,right,0,0,threshold_proximity);
	///arma::cx_mat energies=get<0>(energies_and_dipole_elements);
	///arma::cx_mat dipole_elements=get<2>(energies_and_dipole_elements);
	///cout<<dipole_elements<<endl;
	//arma::vec zeros(3,arma::fill::zeros);
	//for(int w1=0;w1<number_wannier_centers;w1++)
	//	for(int w2=0;w2<number_wannier_centers;w2++){
	//		arma::cx_mat A_ij=function_building_real_space_wannier_dipole_ij(w1,w2,excitonic_momentum,zeros);
	//		cout<<w1<<" "<<w2<<" "<<A_ij<<endl;
	//	}
	arma::field<arma::cx_mat> Mij=function_building_M_k1k2_ij(excitonic_momentum,1);
	cout<<Mij(0)<<endl;
};


/// Dielectric_Function
class Dielectric_Function
{
private:
	int number_k_points_list;
	int number_g_points_list;
	arma::mat g_points_list;
	int number_valence_bands;
	int number_conduction_bands;
	Dipole_Elements *dipole_elements;
	Coulomb_Potential *coulomb_potential;
	int spinorial_calculation;
	double volume_cell;
public:
	Dielectric_Function(Dipole_Elements *dipole_elements_tmp,int number_k_points_list_tmp,int number_g_points_list_tmp,arma::mat g_points_list_tmp,int number_valence_bands_tmp,int number_conduction_bands_tmp,Coulomb_Potential *coulomb_potential_tmp,int spinorial_calculation_tmp,double volume_cell_tmp);
	arma::cx_mat pull_values(arma::vec excitonic_momentum,arma::cx_double omega,double eta,int order_approximation,double threshold_proximity);
	arma::cx_mat pull_values_PPA(arma::vec excitonic_momentum,arma::cx_double omega,double eta,double PPA,int order_approximation,double threshold_proximity);
	void print(arma::vec excitonic_momentum,arma::cx_double omega,double eta,double PPA,int which_term,int order_approximation,double threshold_proximity);
	void pull_macroscopic_value(arma::vec direction,arma::cx_vec omegas_path,int number_omegas_path,double eta,string file_macroscopic_dielectric_function_name,int order_approximation,double threshold_proximity);
	~Dielectric_Function(){
		coulomb_potential=NULL;
		dipole_elements=NULL;
	};
};
Dielectric_Function::Dielectric_Function(Dipole_Elements *dipole_elements_tmp,int number_k_points_list_tmp,int number_g_points_list_tmp,arma::mat g_points_list_tmp,int number_valence_bands_tmp,int number_conduction_bands_tmp,Coulomb_Potential *coulomb_potential_tmp,int spinorial_calculation_tmp,double volume_cell_tmp):
g_points_list(3,number_g_points_list_tmp)
{
	number_conduction_bands=number_conduction_bands_tmp;
	number_valence_bands=number_valence_bands_tmp;
	number_k_points_list=number_k_points_list_tmp;
	number_g_points_list=number_g_points_list_tmp;
	dipole_elements=dipole_elements_tmp;
	coulomb_potential=coulomb_potential_tmp;
	g_points_list=g_points_list_tmp;
	spinorial_calculation=spinorial_calculation_tmp;
	volume_cell=volume_cell_tmp;
};
arma::cx_mat Dielectric_Function::pull_values(arma::vec excitonic_momentum,arma::cx_double omega, double eta, int order_approximation,double threshold_proximity){
	arma::cx_mat epsiloninv(number_g_points_list,number_g_points_list,arma::fill::zeros);
	arma::cx_double ieta; ieta.real(0.0); ieta.imag(eta);
	int number_valence_plus_conduction=number_conduction_bands+number_valence_bands;
	int dimension=(spinorial_calculation+1)*number_k_points_list*number_conduction_bands*number_valence_bands;

	arma::vec zeros_vec(3,arma::fill::zeros);
	std::tuple<arma::cx_mat,arma::cx_mat,arma::cx_mat> energies_rho=dipole_elements->pull_values(excitonic_momentum,zeros_vec,excitonic_momentum,1,0,1,0,0,0,threshold_proximity);
	arma::cx_mat rho_cv=get<2>(energies_rho);
	arma::cx_mat energies=get<0>(energies_rho);
	arma::cx_vec coulomb_shifted(number_g_points_list);
	arma::cx_double ione; ione.real(1.0); ione.imag(0.0);
	int g_point_0=int(number_g_points_list/2);
	
	//auto t1 = std::chrono::high_resolution_clock::now();
	/// defining the denominator factors
	arma::cx_mat rho_reduced_single_column_modified(dimension,number_g_points_list);
	arma::cx_vec multiplicative_factor(dimension);
	for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++){
		//#pragma omp parallel for collapse(3) 
		for(int i=0;i<number_k_points_list;i++)
			for(int c=0;c<number_conduction_bands;c++)
				for(int v=0;v<number_valence_bands;v++)
					multiplicative_factor(spin_channel*number_conduction_bands*number_valence_bands*number_k_points_list+c*number_valence_bands*number_k_points_list+v*number_k_points_list+i)= ione / (omega - energies(spin_channel,c*number_valence_bands*number_k_points_list+v*number_k_points_list+i) + ieta) -  ione / (omega + energies(spin_channel,c*number_valence_bands*number_k_points_list+v*number_k_points_list+i) - ieta);	
	}
	
	///cout<<volume_cell<<endl;
	//#pragma omp parallel for
	for(int i=0;i<number_g_points_list;i++){
		coulomb_shifted(i).real((coulomb_potential->pull(excitonic_momentum+g_points_list.col(i))));
		for(int r=0;r<dimension;r++)
			rho_reduced_single_column_modified(r,i)=rho_cv(r,i)*multiplicative_factor(r);
	}

	const double factor_chi=1/(volume_cell*number_k_points_list);
	arma::cx_double temporary1;
	arma::cx_double temporary2;

	if(order_approximation==0){
		for(int i=0;i<number_g_points_list;i++)
			for(int j=0;j<number_g_points_list;j++){
				temporary1.real(0.0);
				temporary1.imag(0.0);
				for(int r=0;r<dimension;r++)
					temporary1+=conj(rho_cv(r,i))*rho_reduced_single_column_modified(r,j);
				epsiloninv(i,j)=coulomb_shifted(i)*factor_chi*temporary1;
			}
		//#pragma omp parallel for 
		for(int i=0;i<number_g_points_list;i++)
			epsiloninv(i,i).real(epsiloninv(i,i).real()+1.0);
	}else{
		arma::cx_mat epsiloninv_tmp1(number_g_points_list,number_g_points_list,arma::fill::zeros);
		arma::cx_mat chi0(number_g_points_list,number_g_points_list,arma::fill::zeros);
	
		//#pragma omp parallel for collapse(2)
		for(int i=0;i<number_g_points_list;i++)
			for(int j=0;j<number_g_points_list;j++){
				temporary1.real(0.0);
				temporary1.imag(0.0);
				temporary2.real(0.0);
				temporary2.imag(0.0);
				for(int r=0;r<dimension;r++){
					temporary1+=conj(rho_cv(r,i))*rho_reduced_single_column_modified(r,j);
					temporary2+=conj(rho_cv(r,i))*rho_reduced_single_column_modified(r,j);
				}
				epsiloninv_tmp1(i,j)=-(coulomb_shifted(j)*factor_chi)*temporary1;
				chi0(i,j)=factor_chi*temporary2;
			}
		///cout<<"CHI0"<<endl;
		///cout<<chi0(g_point_0,g_point_0)<<endl;
		///cout<<"COULOMB"<<endl;
		///cout<<coulomb_shifted(g_point_0)<<endl;
		///cout<<"CHI0*COULOMB"<<endl;
		///cout<<epsiloninv_tmp1(g_point_0,g_point_0)<<endl;
		
		//#pragma omp parallel for 
		for(int i=0;i<number_g_points_list;i++)
			epsiloninv_tmp1(i,i).real(1.0+epsiloninv_tmp1(i,i).real());
		
		///RPA approximation solving Dyson equation
		epsiloninv=solve(epsiloninv_tmp1,chi0,arma::solve_opts::refine);
		//cout<<"CHI"<<endl;
		///cout<<epsiloninv(g_point_0,g_point_0)<<endl;
		for(int i=0;i<number_g_points_list;i++)
			for(int j=0;j<number_g_points_list;j++)
				epsiloninv(i,j)=epsiloninv(i,j)*coulomb_shifted(i);

		//////#pragma omp parallel for 
		for(int i=0;i<number_g_points_list;i++)
			epsiloninv(i,i).real(epsiloninv(i,i).real()+1.0);
	
	epsiloninv_tmp1.reset();
	chi0.reset();
	}
	///cout<<"EPSILON^-1 RPA"<<endl;
	///cout<<epsiloninv(g_point_0,g_point_0)<<endl;

	multiplicative_factor.reset();
	rho_reduced_single_column_modified.reset();

	return epsiloninv;
};
arma::cx_mat Dielectric_Function::pull_values_PPA(arma::vec excitonic_momentum,arma::cx_double omega,double eta,double PPA,int order_approximation,double threshold_proximity){
	arma::cx_double omega_PPA; omega_PPA.imag(PPA); omega_PPA.real(0.0);
	arma::cx_double omega_0; omega_0.real(0.0); omega_0.imag(0.0);
	arma::cx_double ieta; ieta.real(0.0); ieta.imag(eta);
	
	arma::cx_mat epsiloninv_0=pull_values(excitonic_momentum,omega_0,eta,order_approximation,threshold_proximity);
	arma::cx_mat epsiloninv_PPA=pull_values(excitonic_momentum,omega_PPA,eta,order_approximation,threshold_proximity);
	
	arma::cx_mat rgg(number_g_points_list,number_g_points_list);
	arma::cx_mat ogg(number_g_points_list,number_g_points_list);
	
	ogg=PPA*sqrt(epsiloninv_PPA/(epsiloninv_0-epsiloninv_PPA));
	rgg=(epsiloninv_0%ogg)/2;
	
	arma::cx_mat epsilon_app(number_g_points_list,number_g_points_list,arma::fill::zeros);

	for(int i=0;i<number_g_points_list;i++)
		for(int j=0;j<number_g_points_list;j++)
			epsilon_app(i,j)=rgg(i,j)*(1.0/(omega-ogg(i,j)+ieta)-1.0/(omega+ogg(i,j)-ieta));
	for(int i=0;i<number_g_points_list;i++)
		epsilon_app(i,i)+=1.0;
	return epsilon_app;
};
void Dielectric_Function::pull_macroscopic_value(arma::vec direction,arma::cx_vec omegas_path,int number_omegas_path,double eta,string file_macroscopic_dielectric_function_name,int order_approximation,double threshold_proximity){
	///THIS SOLUTION IS UNSTABLE
	arma::cx_mat macroscopic_dielectric_function(number_g_points_list,number_g_points_list);
	arma::cx_mat macroscopic_dielectric_function_inv(number_g_points_list,number_g_points_list);
	arma::cx_vec ones(number_g_points_list,arma::fill::ones);
	arma::vec q_point_0(3,arma::fill::zeros);
	arma::cx_double ieta; ieta.real(0.0); ieta.imag(eta);
	q_point_0(0)+=minval;
	int g_point_0=int(number_g_points_list/2);
	ofstream file_macroscopic_dielectric_function;
	file_macroscopic_dielectric_function.open(file_macroscopic_dielectric_function_name);
	for(int i=0;i<number_omegas_path;i++){
		macroscopic_dielectric_function_inv=pull_values(q_point_0,omegas_path(i),eta,order_approximation,threshold_proximity);
		macroscopic_dielectric_function=solve(macroscopic_dielectric_function_inv,diagmat(ones));
		///macroscopic_dielectric_function(g_point_0,g_point_0).imag(macroscopic_dielectric_function(g_point_0,g_point_0).imag()*100);
		cout<<i<<" "<<number_omegas_path<<macroscopic_dielectric_function(g_point_0,g_point_0)<<endl;
		file_macroscopic_dielectric_function<<i<<" "<<omegas_path(i)<<" "<<macroscopic_dielectric_function(g_point_0,g_point_0)<<endl;
	}
	//CONSIDERING THE SOLUTION OF ANALYTICAL EXPANSION
	//int g_point_0=int(number_g_points_list/2);
	//cx_mat vchi(number_g_points_list,number_g_points_list,fill::zeros);
	//cx_double macroscopic_epsilon;
	//cx_double ieta; ieta.real(0.0); ieta.imag(eta);
	//int number_valence_plus_conduction=number_conduction_bands+number_valence_bands;
	//vec zeros_vec(3,fill::zeros);
	//tuple<cx_mat,cx_mat> energies_rho=dipole_elements->pull_values(direction,zeros_vec,direction,1,0,0);
	//cx_mat rho_cv=dipole_elements->pull_reduced_values_cv(get<1>(energies_rho),1,0);
	//cx_mat energies=get<0>(energies_rho);
	//cx_vec coulomb_shifted(number_g_points_list);
	//cx_double ione; ione.real(1.0); ione.imag(0.0);
	//cx_mat rho_reduced_single_column_modified((spinorial_calculation+1)*number_k_points_list*number_conduction_bands*number_valence_bands,number_g_points_list);
	//cx_vec multiplicative_factor((spinorial_calculation+1)*number_k_points_list*number_conduction_bands*number_valence_bands);
	//double factor_chi=4*pigreco/(volume_cell*number_k_points_list);
	//ofstream file_macroscopic_dielectric_function;
	//file_macroscopic_dielectric_function.open(file_macroscopic_dielectric_function_name);
	//for(int s=0;s<number_omegas_path;s++){
	//	for(int spin_channel=0;spin_channel<(spinorial_calculation+1);spin_channel++){
	//		#pragma omp parallel for collapse(3) 
	//		for(int i=0;i<number_k_points_list;i++)
	//			for(int c=0;c<number_conduction_bands;c++)
	//				for(int v=0;v<number_valence_bands;v++)
	//					multiplicative_factor(spin_channel*number_conduction_bands*number_valence_bands*number_k_points_list+c*number_valence_bands*number_k_points_list+v*number_k_points_list+i)= ione / (omegas_path(s) - energies(spin_channel,c*number_valence_bands*number_k_points_list+v*number_k_points_list+i) + ieta) -  ione / (omega(s) + energies(spin_channel,c*number_valence_bands*number_k_points_list+v*number_k_points_list+i) - ieta);
	//	}
	//	///cout<<volume_cell<<endl;
	//	#pragma omp parallel for
	//	for(int i=0;i<number_g_points_list;i++){
	//		coulomb_shifted(i).real((coulomb_potential->pull(direction+g_points_list.col(i))));
	//		rho_reduced_single_column_modified.col(i)=rho_cv.col(i)%multiplicative_factor;
	//	}
	///	if(order_approximation==0){
	///		#pragma omp parallel for collapse(2) 
	///		for(int i=0;i<number_g_points_list;i++)
	///			for(int j=0;j<number_g_points_list;j++)
	///				vchi(i,j)=-coulomb_shifted(i)*factor_chi*accu(conj(rho_cv.col(i))%rho_reduced_single_column_modified.col(j));
	///		#pragma omp parallel for 
	///		for(int i=0;i<number_g_points_list;i++)
	///			vchi(i,i).real(-vchi(i,i).real()+1.0);
	///	
	///		macroscopic_epsilon=vchi(g_point_0,g_point_0);
	///		for(int k=0;k<number_g_points_list;k++)
	///			if(k!=g_point_0)
	///				macroscopic_epsilon+=vchi(g_point_0,k)*vchi(k,g_point_0);
	///	}else{
	///	}
	///	
	///	cout<<i<<" "<<number_omegas_path<<<<endl;
	///	file_macroscopic_dielectric_function<<i<<" "<<omegas_path(i)<<" "<<macroscopic_dielectric_function(g_point_0,g_point_0)<<endl;
	///file_macroscopic_dielectric_function.close();
	///vchi.reset();
	///multiplicative_factor.reset();
	///rho_reduced_single_column_modified.reset();
	///return 
};
void Dielectric_Function::print(arma::vec excitonic_momentum,arma::cx_double omega,double eta,double PPA,int which_term,int order_approximation,double threshold_proximity){
	arma::cx_mat dielectric_function(number_g_points_list,number_g_points_list);
	if(which_term==0)
		dielectric_function=pull_values(excitonic_momentum,omega,eta,order_approximation,threshold_proximity);
	else
		dielectric_function=pull_values_PPA(excitonic_momentum,omega,eta,PPA,order_approximation,threshold_proximity);
	
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
	int spin_dimension_bse_hamiltonian_4;
	int spin_dimension_bse_hamiltonian_2;
	int spin_dimension_bse_hamiltonian_2_frac_tdf;
	int spin_dimension_bse_hamiltonian_2_mult_tdf;
	int spin_dimension_bse_hamiltonian_4_frac_tdf;
	int spin_dimension_bse_hamiltonian_4_mult_tdf;
	int htb_basis_dimension;
	int number_k_points_list;
	int number_valence_times_conduction;
	int number_g_points_list;
	int insulator_metal;
	double cell_volume;
	arma::mat bravais_lattice{arma::mat(3,3)};
	arma::mat exciton_spin{arma::mat(2,4)};
	arma::vec excitonic_momentum{arma::vec(3)};
	arma::mat k_points_list;
	arma::mat g_points_list;
	arma::mat exciton;
	int tamn_dancoff;
	double threshold_proximity;
	Dipole_Elements *dipole_elements;
	arma::cx_cube v_coulomb_gg;
	arma::cx_vec v_coulomb_g;
	arma::cx_mat excitonic_hamiltonian;
	arma::cx_mat rho_q_diagk_cv;

	arma::mat k_points_differences;
public:
	/// be carefull: do not try to build the BSE matrix with more bands than those given by the hamiltonian!!!
	/// there is a check at the TB hamiltonian level but not here...
	Excitonic_Hamiltonian(int number_valence_bands_tmp,int number_conduction_bands_tmp, arma::mat k_points_list_tmp, int number_k_points_list_tmp, arma::mat g_points_list_tmp,int number_g_points_list_tmp, int spinorial_calculation_tmp, int htb_basis_dimension_tmp,Dipole_Elements *dipole_elements_tmp, double cell_volume_tmp,int tamn_dancoff_tmp,int insulator_metal_tmp,arma::mat k_points_differences_tmp,double threshold_proximity_tmp);
	void pull_coulomb_potentials(Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,arma::vec excitonic_momentum,double eta,int order_approximation,int number_integration_points,int reading_W,int adding_momentum);
	void pull_resonant_part_and_rcv(arma::vec excitonic_momentum_tmp,int ipa);
	void add_coupling_part();
	std::tuple<arma::cx_mat,arma::cx_mat> extract_hbse_and_rcv(arma::vec excitonic_momentum_tmp,double eta,Coulomb_Potential *coulomb_potential,Dielectric_Function *dielectric_function,int adding_screening,int tamn_dancoff,int order_approximation,int number_integration_points,int reading_W,int ipa);
	std::tuple<arma::cx_vec,arma::cx_mat> common_diagonalization(int ipa);
	///tuple<vec,cx_mat> cholesky_diagonalization(double eta);
	std::tuple<arma::cx_vec,arma::cx_vec> pull_excitonic_oscillator_force(arma::cx_mat excitonic_eigenstates,int tamn_dancoff,int ipa);
	void pull_macroscopic_bse_dielectric_function(arma::cx_vec omegas_path,int number_omegas_path,double eta,string file_macroscopic_dielectric_function_bse_name,double lorentzian,int tamn_dancoff,Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,int order_approximation,int number_integration_points,int reading_W,int ipa);
	void print(arma::vec excitonic_momentum_tmp,double eta,int tamn_dancoff,Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,int order_approximation,int number_integration_points,int reading_W,int ipa);
	arma::cx_mat pull_augmentation_matrix(arma::cx_mat exc_eigenstates,int spin_dimension_bse_hamiltonian_4_frac_tdf);
	void spin_transformation();
	///~Excitonic_Hamiltonian(){
	///	v_coulomb_gg.reset();
	///	v_coulomb_g.reset();
	///	excitonic_hamiltonian.reset();
	///	rho_q_diagk_cv.reset();
	///	rho_p_diagk_vc.reset();
	///}
};
Excitonic_Hamiltonian::Excitonic_Hamiltonian(int number_valence_bands_tmp,int number_conduction_bands_tmp, arma::mat k_points_list_tmp, int number_k_points_list_tmp, arma::mat g_points_list_tmp,int number_g_points_list_tmp,int spinorial_calculation_tmp,int htb_basis_dimension_tmp,Dipole_Elements *dipole_elements_tmp,double cell_volume_tmp,int tamn_dancoff_tmp,int insulator_metal_tmp,arma::mat k_points_differences_tmp,double threshold_proximity_tmp):
k_points_differences(3,number_k_points_list_tmp*number_k_points_list_tmp),k_points_list(3,number_k_points_list_tmp),g_points_list(3,number_g_points_list_tmp),exciton(2, number_valence_bands_tmp*number_conduction_bands_tmp),v_coulomb_gg(number_g_points_list_tmp,number_g_points_list_tmp,number_k_points_list_tmp*number_k_points_list_tmp),
v_coulomb_g(number_g_points_list_tmp),excitonic_hamiltonian((2-tamn_dancoff_tmp)*(3*spinorial_calculation_tmp+1)*number_conduction_bands_tmp*number_valence_bands_tmp*number_k_points_list_tmp,(2-tamn_dancoff_tmp)*(3*spinorial_calculation_tmp+1)*number_conduction_bands_tmp*number_valence_bands_tmp*number_k_points_list_tmp),
rho_q_diagk_cv((2-tamn_dancoff_tmp)*(spinorial_calculation_tmp+1)*number_conduction_bands_tmp*number_valence_bands_tmp*number_k_points_list_tmp,number_g_points_list_tmp)
{
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

	bravais_lattice=dipole_elements_tmp->pull_bravais_lattice();

	threshold_proximity=threshold_proximity_tmp;
	k_points_differences=k_points_differences_tmp;
	tamn_dancoff=tamn_dancoff_tmp;

	cell_volume=cell_volume_tmp;
	dipole_elements=dipole_elements_tmp;
	insulator_metal=insulator_metal_tmp;
	
	int e = 0;
	for (int v = 0; v < number_valence_bands; v++)
		for (int c = 0; c < number_conduction_bands; c++){
			exciton(0, e) = c;
			exciton(1, e) = v;
			e++;
		}
	cout<<exciton<<endl;
	/// defining the possible spin combinations
	cout<<size(exciton_spin)<<endl;
	exciton_spin.zeros();
	exciton_spin(1, 1) = 1;
	exciton_spin(0, 2) = 1;
	exciton_spin(0, 3) = 1;
	exciton_spin(1, 3) = 1;
	
	spin_dimension_bse_hamiltonian_2=(spinorial_calculation+1)*dimension_bse_hamiltonian;
	spin_dimension_bse_hamiltonian_4=(3*spinorial_calculation+1)*dimension_bse_hamiltonian;
	spin_dimension_bse_hamiltonian_4_frac_tdf=spin_dimension_bse_hamiltonian_4/(1+tamn_dancoff);
	spin_dimension_bse_hamiltonian_2_frac_tdf=spin_dimension_bse_hamiltonian_2/(1+tamn_dancoff);
	spin_dimension_bse_hamiltonian_4_mult_tdf=(2-tamn_dancoff)*spin_dimension_bse_hamiltonian_4;	
	spin_dimension_bse_hamiltonian_2_mult_tdf=(2-tamn_dancoff)*spin_dimension_bse_hamiltonian_2;	

	///cout<<"Allocating HBSE memory (and rho memory)"<<endl;
	//excitonic_hamiltonian(spin_dimension_bse_hamiltonian_4_mult_tdf,spin_dimension_bse_hamiltonian_4_mult_tdf){};
	//rho_q_diagk_cv(spin_dimension_bse_hamiltonian_2_mult_tdf,number_g_points_list){};
	//v_coulomb_gg(number_g_points_list,number_g_points_list,number_k_points_list*number_k_points_list){};
	//v_coulomb_g(number_g_points_list){};
	///cout<<size(rho_q_diagk_cv)<<endl;
	///rho_p_diagk_vc.set_size(number_g_points_list,(spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list);
	///cout<<"Finished allocating HBSE memory"<<endl;

};
/// calculating the potentianl before the BSE hamiltonian building
/// calculating the generalized potential (the screened one and the unscreened-one)
void Excitonic_Hamiltonian::pull_coulomb_potentials(Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,arma::vec excitonic_momentum,double eta,int order_approximation,int number_integration_points,int reading_W,int adding_momentum){
	
	if(adding_momentum==1){
		for(int i=0;i<number_k_points_list*number_k_points_list;i++)
			for(int r=0;r<3;r++)
				k_points_differences(r,i)=k_points_differences(r,i)-excitonic_momentum(r);
		for (int k = 0; k < number_g_points_list; k++)
			v_coulomb_g(k)=0.0;
		for(int i = 0; i < number_k_points_list; i++)
			for(int j = 0; j < number_k_points_list; j++)
				for (int k = 0; k < number_g_points_list; k++)
					for (int s = 0; s < number_g_points_list; s++)
						v_coulomb_gg(k,s,i*number_k_points_list+j)=0.0;
	}

	arma::cx_double omega_0; omega_0.real(0.0); omega_0.imag(0.0);
	arma::cx_mat epsilon_inv_static(number_g_points_list,number_g_points_list);
	
	int g_point_0=int(number_g_points_list/2);
	for (int k = 0; k < number_g_points_list; k++)
		v_coulomb_g(k) = coulomb_potential->pull(excitonic_momentum+g_points_list.col(k));

	int writing_on_file_W=1;

	if(reading_W==0){
		arma::vec k_point(3);
		if(insulator_metal==1){
			arma::cx_mat temporary_matrix(number_g_points_list,number_g_points_list);
			//cout<<" 1"<<endl;
			if(adding_screening==1){
				for(int i = 0; i < number_k_points_list; i++)
					for(int j = 0; j < number_k_points_list; j++){
						k_point=k_points_differences.col(i*number_k_points_list+j);
						temporary_matrix=dielectric_function->pull_values(k_point,omega_0,eta,order_approximation,threshold_proximity);
						for (int k = 0; k < number_g_points_list; k++)
							for (int s = 0; s < number_g_points_list; s++)
								v_coulomb_gg(k,s,i*number_k_points_list+j)=temporary_matrix(k,s)*coulomb_potential->pull(k_points_differences.col(i*number_k_points_list+j)+g_points_list.col(s));
					}
			}else{
				temporary_matrix.eye();
				for(int i = 0; i < number_k_points_list; i++)
					for(int j = 0; j < number_k_points_list; j++)
						for (int k = 0; k < number_g_points_list; k++)
							for (int s = 0; s < number_g_points_list; s++)
								v_coulomb_gg(k,s,i*number_k_points_list+j)=temporary_matrix(k,s)*coulomb_potential->pull(k_points_differences.col(i*number_k_points_list+j)+g_points_list.col(s));
			}			
		}else{
			///in the case of an insulator 0
			///epsilon for k-k'=0 is not sufficient to compensate the diverging coulomb
			///so for these points an average value is considered
			///the average is obtained through an integration over a little sphere
			///this is the sphere over which is the average of W is evaluated (this is a way to cure its divergence)
			double radius=2*pigreco*std::pow((3/(4*pigreco*number_k_points_list*cell_volume)),1/3);	
			cout<<"building "<<endl;
			cout<<"differentiating between k points "<<endl;
			int counting_gt0=0;
			int counting_0=0;
			for(int i = 0; i < number_k_points_list; i++)
				for(int j = 0; j < number_k_points_list; j++){
					if(norm(k_points_differences.col(i*number_k_points_list+j))>minval)
						counting_gt0+=1;
					else
						counting_0+=1;
				}
			arma::vec k_points_differences_gt0(counting_gt0);
			arma::vec k_points_differences_0(counting_0);
			counting_0=0;
			counting_gt0=0;
			for(int i = 0; i < number_k_points_list; i++)
				for(int j = 0; j < number_k_points_list; j++){
					if(norm(k_points_differences.col(i*number_k_points_list+j))>minval){
						k_points_differences_gt0(counting_gt0)=i*number_k_points_list+j;
						counting_gt0+=1;
					}else{
						k_points_differences_0(counting_0)=i*number_k_points_list+j;
						counting_0+=1;
					}
				}

			double coulomb_potential_average_g1;
			double coulomb_potential_average_g2;
			double coulomb_potential_average_g1_0=conversion_parameter*4*pigreco*radius*number_k_points_list*cell_volume/pow(2*pigreco,3);
			///TEST
			///double coulomb_potential_average_g1_0=0.0;
			///cout<<adding_screening<<endl;
			if(adding_screening==1){
				cout<<"building dielectric function "<<counting_0<<" "<<counting_gt0<<endl;
				int counting=0;
				arma::mat list_temporary_vectors(3,number_integration_points);
				arma::vec temporary_vector(3);
				std::random_device rd;
    			std::mt19937 gen(rd());
   				std::uniform_real_distribution<> dis(0.0,1.0); // distribution in range [0,1]
				while(counting<number_integration_points){
					///cout<<counting<<"MONTE-CARLO INTEGRATION  "<<endl;
					for(int i=0;i<3;i++){
						if(dis(gen)>0.5){
							temporary_vector(i)=-dis(gen)*radius;
						}else
							temporary_vector(i)=dis(gen)*radius;
					}
					////inside the sphere
					if((norm(temporary_vector,2)<=radius)&&(norm(temporary_vector,2)>minval*minval)){
						list_temporary_vectors.col(counting)=temporary_vector;
						counting+=1;
					}
				}
				arma::cx_mat average_inv_epsilon(number_g_points_list,number_g_points_list);
				cout<<"averaging dielectric function around 0"<<endl;
				for (int i=0; i<number_integration_points; i++)
					average_inv_epsilon+=dielectric_function->pull_values(list_temporary_vectors.col(i),omega_0,eta,order_approximation,threshold_proximity);

				for (int s = 0; s < number_g_points_list; s++)
					for (int k = 0; k < number_g_points_list; k++){
						average_inv_epsilon(s,k).real(average_inv_epsilon(s,k).real()/number_integration_points);
						average_inv_epsilon(s,k).imag(average_inv_epsilon(s,k).imag()/number_integration_points);
					}		

				cout<<"building W function taking into account diverging points"<<endl;
				arma::cx_cube temporary_matrix(number_g_points_list,number_g_points_list,number_k_points_list*number_k_points_list);
				for(int c = 0; c < counting_gt0; c++)	
					temporary_matrix.subcube(0,0,k_points_differences_gt0(c),number_g_points_list-1,number_g_points_list-1,k_points_differences_gt0(c))=arma::cx_mat(dielectric_function->pull_values(k_points_differences.col(k_points_differences_gt0(c)),omega_0,eta,order_approximation,threshold_proximity));
				for(int c = 0; c < counting_0; c++)
					temporary_matrix.subcube(0,0,k_points_differences_0(c),number_g_points_list-1,number_g_points_list-1,k_points_differences_0(c))=arma::cx_mat(average_inv_epsilon);
			
				//#pragma omp parallel for collapse(3)
				for(int c = 0; c < counting_gt0; c++)
					for (int s = 0; s < number_g_points_list; s++)
						for (int k = 0; k < number_g_points_list; k++){								
							coulomb_potential_average_g1=sqrt(coulomb_potential->pull(k_points_differences.col(k_points_differences_gt0(c))+g_points_list.col(s)));
							coulomb_potential_average_g2=sqrt(coulomb_potential->pull(k_points_differences.col(k_points_differences_gt0(c))+g_points_list.col(k)));
							v_coulomb_gg(k,s,k_points_differences_gt0(c))=temporary_matrix(k,s,k_points_differences_gt0(c))*coulomb_potential_average_g1*coulomb_potential_average_g2;
						}
				//#pragma omp parallel for collapse(3)
				for(int c = 0; c < counting_0; c++)
					for (int s = 0; s < number_g_points_list; s++)
						for (int k = 0; k < number_g_points_list; k++)		
							v_coulomb_gg(k,s,k_points_differences_0(c))=temporary_matrix(k,s,k_points_differences_0(c))*coulomb_potential_average_g1_0;

				if(writing_on_file_W==1){
					ofstream w_coulomb_potential_file;
					w_coulomb_potential_file.open("W_coulomb_potential_file.data");
					for(int i = 0; i < number_k_points_list; i++)
						for(int j = 0; j < number_k_points_list; j++)
							for (int s = 0; s < number_g_points_list; s++)
								for (int k = 0; k < number_g_points_list; k++)		
									w_coulomb_potential_file<<v_coulomb_gg(k,s,i*number_k_points_list+j)<<"	";
					w_coulomb_potential_file.close();
				}

			}else{
				double empirical_inv_epsilon=0.084033613;
				////double empirical_inv_epsilon=0.08264462809917356;

				///double empirical_inv_epsilon=0.1;
				///#pragma omp parallel for collapse(3)
				for(int c = 0; c < counting_gt0; c++)
					for (int s = 0; s < number_g_points_list; s++)
						for (int k = 0; k < number_g_points_list; k++){								
							if(k==s){
								coulomb_potential_average_g1=sqrt(coulomb_potential->pull(k_points_differences.col(k_points_differences_gt0(c))+g_points_list.col(s)));
								coulomb_potential_average_g2=sqrt(coulomb_potential->pull(k_points_differences.col(k_points_differences_gt0(c))+g_points_list.col(k)));
								v_coulomb_gg(k,s,k_points_differences_gt0(c))=empirical_inv_epsilon*coulomb_potential_average_g1*coulomb_potential_average_g2;
							}else{
								v_coulomb_gg(k,s,k_points_differences_gt0(c)).real(0.0);
								v_coulomb_gg(k,s,k_points_differences_gt0(c)).imag(0.0);
							}
						}
				//	cout<<v_coulomb_gg.slice(k_points_differences_gt0(c))<<endl;

				///#pragma omp parallel for collapse(3)
				for(int c = 0; c < counting_0; c++)
					for (int s = 0; s < number_g_points_list; s++)
						for (int k = 0; k < number_g_points_list; k++){	
							if(k==s){
								v_coulomb_gg(s,s,k_points_differences_0(c))=empirical_inv_epsilon*coulomb_potential_average_g1_0;
							}else{
								v_coulomb_gg(k,s,k_points_differences_0(c)).real(0.0);
								v_coulomb_gg(k,s,k_points_differences_0(c)).imag(0.0);
							}						
						}
			}
		}
	}else{
		ifstream w_coulomb_potential_file;
		w_coulomb_potential_file.open("W_coulomb_potential_file.data");
		w_coulomb_potential_file.seekg(0);
		while (w_coulomb_potential_file.peek() != EOF){
			for(int i = 0; i < number_k_points_list; i++)
				for(int j = 0; j < number_k_points_list; j++)
					for (int s = 0; s < number_g_points_list; s++)
						for (int k = 0; k < number_g_points_list; k++)		
							w_coulomb_potential_file>>v_coulomb_gg(k,s,i*number_k_points_list+j);		
		}
		w_coulomb_potential_file.close();
	}
		///for(int i = 0; i < number_k_points_list; i++)
			///	for(int j = 0; j < number_k_points_list; j++)
			///		cout<<v_coulomb_gg(0,0,i*number_k_points_list+j)<<" ";
			///UNSCREENING PART TO BE IMPLEMENTED
			///for (int k = 0; k < number_g_points_list; k++)
			///	for (int s = 0; s < number_g_points_list; s++){
			///		v_coulomb_gg(k,s).real(number_k_points_list*cell_volume*v_coulomb_gg(k,s).real()/(number_integration_points*pow(2*pigreco,3)));
		///		v_coulomb_gg(k,s).imag(number_k_points_list*cell_volume*v_coulomb_gg(k,s).imag()/(number_integration_points*pow(2*pigreco,3)));
		///	}
	///cout<<" W"<<endl;
	///for (int k = 0; k < number_g_points_list; k++)
	///	for (int s = 0; s < number_g_points_list; s++)
	///		cout<<v_coulomb_gg(k,s)<<" ";
	///cout<<"LONG RANGE PART W"<<endl;
	
	cout<<"W00 "<<v_coulomb_gg(g_point_0,g_point_0,0)<<endl;
	///cout<<"W11 "<<v_coulomb_gg<<endl;
	///cout<<"V00 "<<v_coulomb_g(g_point_0)<<endl;
	///cout<<"V11 "<<v_coulomb_g(0)<<endl;
	///cout<<"V coulomb no LONG-RANGE PART"<<endl;
	//for(int i=0;i<number_g_points_list;i++)
	//	if(i!=g_point_0)
	//	cout<<v_coulomb_g(i)<<" ";
};
void Excitonic_Hamiltonian::pull_resonant_part_and_rcv(arma::vec excitonic_momentum_tmp,int ipa){
	//cleaning hamiltonian 
	for(int i=0;i<spin_dimension_bse_hamiltonian_4_mult_tdf;i++)
		for(int j=0;j<spin_dimension_bse_hamiltonian_4_mult_tdf;j++){
			excitonic_hamiltonian(i,j).real(0.0);
			excitonic_hamiltonian(i,j).imag(0.0);
		}
	for(int i=0;i<spin_dimension_bse_hamiltonian_2_mult_tdf;i++)
		for(int g=0;g<number_g_points_list;g++){
			rho_q_diagk_cv(i,g).real(0.0);
			rho_q_diagk_cv(i,g).imag(0.0);
		}
	//cout<<excitonic_hamiltonian<<endl;
			
	excitonic_momentum=excitonic_momentum_tmp;
	///BEOFRE BUILDING THE BSE HAMILTONIAN, BE SURE TO HAVE BUILT THE POTENTIALS (pull_coulomb_potentials)
	///building entry 00 (resonant part)
	cout<<"building 00"<<endl;
	///building v
	//cx_mat v_matrix(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian,fill::zeros);
	cout<<"building v"<<endl;
	cout<<"beginning extracting rho"<<endl;
	arma::vec zeros_vec(3,arma::fill::zeros);
	std::tuple<arma::cx_mat,arma::cx_mat,arma::cx_mat> energies_rho_q_diagk_cv=dipole_elements->pull_values(excitonic_momentum,zeros_vec,excitonic_momentum,1,0,1,0,0,0,threshold_proximity);
	rho_q_diagk_cv.submat(0,0,spin_dimension_bse_hamiltonian_2-1,number_g_points_list-1)=get<2>(energies_rho_q_diagk_cv);
	arma::cx_mat energies_q_diff=get<0>(energies_rho_q_diagk_cv);
	arma::cx_mat energies_q_sum=get<1>(energies_rho_q_diagk_cv);
	///cout<<rho_q_diagk_cv<<endl;
	if(ipa==0){
		cout<<"ending extracting rho"<<endl;
		arma::cx_vec zeros_long_vec(spin_dimension_bse_hamiltonian_2,arma::fill::zeros);
		arma::cx_mat temporary_matrix1(number_g_points_list,spin_dimension_bse_hamiltonian_2,arma::fill::zeros);
		arma::cx_double factor_v;
		factor_v.real((2-spinorial_calculation)/(number_k_points_list*cell_volume));
		factor_v.imag(0.0);

		for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++)
			for(int r=0;r<number_g_points_list;r++){
				temporary_matrix1(r,i).real(real(factor_v*(v_coulomb_g(r)*conj(rho_q_diagk_cv(i,r)))));
				temporary_matrix1(r,i).imag(imag(factor_v*(v_coulomb_g(r)*conj(rho_q_diagk_cv(i,r)))));
			}
		////excluding G=0 
		cout<<"second part"<<endl;
		int g_point_0=int(number_g_points_list/2);
		temporary_matrix1.row(g_point_0)=zeros_long_vec.t();
		////cout<<rho_q_diagk_cv<<endl;
		arma::cx_mat rho_q_diagk_cv_tmp(number_g_points_list,spin_dimension_bse_hamiltonian_2);
		rho_q_diagk_cv_tmp=(rho_q_diagk_cv.submat(0,0,spin_dimension_bse_hamiltonian_2-1,number_g_points_list-1)).t();
		rho_q_diagk_cv_tmp.row(g_point_0)=zeros_long_vec.t();
		///#pragma omp parallel for collapse(6)
		for(int i=0;i<(spinorial_calculation+1);i++)
			for(int j=0;j<(spinorial_calculation+1);j++)
				for(int r=0;r<dimension_bse_hamiltonian;r++)
					for(int s=0;s<dimension_bse_hamiltonian;s++)
						for(int g=0;g<number_g_points_list;g++){
							excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,j*3*number_conduction_bands*number_valence_bands*number_k_points_list+s).real(excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,j*3*number_conduction_bands*number_valence_bands*number_k_points_list+s).real()+real(temporary_matrix1(g,i*number_conduction_bands*number_valence_bands*number_k_points_list+r)*(rho_q_diagk_cv_tmp(g,j*number_conduction_bands*number_valence_bands*number_k_points_list+s))));
							excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,j*3*number_conduction_bands*number_valence_bands*number_k_points_list+s).imag(excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,j*3*number_conduction_bands*number_valence_bands*number_k_points_list+s).imag()+imag(temporary_matrix1(g,i*number_conduction_bands*number_valence_bands*number_k_points_list+r)*(rho_q_diagk_cv_tmp(g,j*number_conduction_bands*number_valence_bands*number_k_points_list+s))));
						}
		cout<<"third part"<<endl;
		rho_q_diagk_cv_tmp.reset();
		temporary_matrix1.reset();
		zeros_long_vec.reset();
		///cout<<excitonic_hamiltonian<<endl;
		///building w
		
		cout<<"building w"<<endl;
		cout<<"beginning extracting rho"<<endl;
		arma::cx_mat rho_kk_cc=get<2>(dipole_elements->pull_values(zeros_vec,zeros_vec,zeros_vec,0,0,1,1,0,0,threshold_proximity));
		arma::cx_mat rho_qq_kk_vv=get<2>(dipole_elements->pull_values(zeros_vec,excitonic_momentum,excitonic_momentum,0,0,0,0,0,0,threshold_proximity));
		///cout<<rho_kk_cc<<endl;
		///cout<<rho_qq_kk_vv<<endl;
		///cout<<v_coulomb_gg<<endl;
		cout<<"ending extracting rho"<<endl;
		cout<<"fourth part"<<endl;
		arma::cx_mat v_coulomb_temp(number_g_points_list,number_g_points_list,arma::fill::zeros);
		arma::cx_mat temporary_matrix2(number_k_points_list,number_g_points_list);
		arma::cx_vec temporary_vector2(number_conduction_bands*number_valence_bands*number_k_points_list);
		arma::cx_double factor_w;
		factor_w.real(-1.0/(cell_volume*number_k_points_list));
		//factor_w.real(1.0);
		factor_w.imag(0.0);
		int spinv1; int spinc1;
		for(int spin1=0;spin1<(3*spinorial_calculation+1);spin1++){
			spinv1=exciton_spin(0,spin1);
			spinc1=exciton_spin(1,spin1);
			for(int c1=0;c1<number_conduction_bands;c1++)
				for(int v1=0;v1<number_valence_bands;v1++)
					for(int k1=0;k1<number_k_points_list;k1++){
						for(int c2=0;c2<number_conduction_bands;c2++){
							for(int k2=0;k2<number_k_points_list;k2++)
								for(int s=0;s<number_g_points_list;s++){
									temporary_matrix2(k2,s).real(0.0);
									temporary_matrix2(k2,s).imag(0.0);
									for(int g=0;g<number_g_points_list;g++){
										temporary_matrix2(k2,s).real(temporary_matrix2(k2,s).real()+real(conj(rho_kk_cc(spinc1*number_conduction_bands*number_conduction_bands*number_k_points_list*number_k_points_list+c2*number_conduction_bands*number_k_points_list*number_k_points_list+c1*number_k_points_list*number_k_points_list+k2*number_k_points_list+k1,g))*(v_coulomb_gg(g,s,k2*number_k_points_list+k1))));
										temporary_matrix2(k2,s).imag(temporary_matrix2(k2,s).imag()+imag(conj(rho_kk_cc(spinc1*number_conduction_bands*number_conduction_bands*number_k_points_list*number_k_points_list+c2*number_conduction_bands*number_k_points_list*number_k_points_list+c1*number_k_points_list*number_k_points_list+k2*number_k_points_list+k1,g))*(v_coulomb_gg(g,s,k2*number_k_points_list+k1))));
									}
								}
							for(int v2=0;v2<number_valence_bands;v2++)
								for(int k2=0;k2<number_k_points_list;k2++){
									temporary_vector2(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).real(0.0);
									temporary_vector2(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).imag(0.0);
									for(int g=0;g<number_g_points_list;g++){
										temporary_vector2(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).real(temporary_vector2(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).real()+real(factor_w*temporary_matrix2(k2,g)*(rho_qq_kk_vv(spinv1*number_valence_bands*number_valence_bands*number_k_points_list*number_k_points_list+v2*number_valence_bands*number_k_points_list*number_k_points_list+v1*number_k_points_list*number_k_points_list+k2*number_k_points_list+k1,g))));
										temporary_vector2(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).imag(temporary_vector2(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).imag()+imag(factor_w*temporary_matrix2(k2,g)*(rho_qq_kk_vv(spinv1*number_valence_bands*number_valence_bands*number_k_points_list*number_k_points_list+v2*number_valence_bands*number_k_points_list*number_k_points_list+v1*number_k_points_list*number_k_points_list+k2*number_k_points_list+k1,g))));
									}
								}
						}
						excitonic_hamiltonian.submat(spin1*number_conduction_bands*number_valence_bands*number_k_points_list+c1*number_valence_bands*number_k_points_list+v1*number_k_points_list+k1,
							spin1*number_conduction_bands*number_valence_bands*number_k_points_list,spin1*number_conduction_bands*number_valence_bands*number_k_points_list+c1*number_valence_bands*number_k_points_list+v1*number_k_points_list+k1,
							(spin1+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1)=temporary_vector2.t();
					}
		}
		rho_kk_cc.reset();
		rho_qq_kk_vv.reset();
		temporary_matrix2.reset();
		temporary_vector2.reset();
	}
	///cout<<excitonic_hamiltonian<<endl;
	///separated == 1 gives H00=Resonant Part H11=Coupling Part
	///rewriting energies in order to obtain something that can be summed to the rest
	cout<<"fifth part"<<endl;
	///cout<<excitonic_hamiltonian<<endl;
	//cout<<"ecco "<<energies_q_vc<<endl;
	//#pragma omp parallel for collapse(2) 
	for(int i=0;i<(spinorial_calculation+1);i++)
		for(int r=0;r<number_conduction_bands*number_valence_bands*number_k_points_list;r++){
			excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r).real(real(excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r))+real(energies_q_diff(i,r)));
			excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r).imag(imag(excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r))+imag(energies_q_diff(i,r)));
		}
	if(spinorial_calculation==1){
		arma::cx_mat cv_up_down(spinorial_calculation+1,number_conduction_bands*number_valence_bands*number_k_points_list);
		arma::cx_double two; two.real(2.0); two.imag(0.0);
		//#pragma omp parallel for collapse(4) 
		for(int i=0;i<(spinorial_calculation+1);i++)
			for(int c=0;c<number_conduction_bands;c++)
				for(int v=0;v<number_valence_bands;v++)
					for(int k=0;k<number_k_points_list;k++)
						cv_up_down(i,c*number_valence_bands*number_k_points_list+v*number_k_points_list+k)
							=((energies_q_diff(1-i,c*number_k_points_list+k)+energies_q_sum(1-i,c*number_k_points_list+k))
								-(-energies_q_diff(i,v*number_k_points_list+k)+energies_q_sum(i,v*number_k_points_list+k)))/two;
		///#pragma omp parallel for collapse(2) 
		for(int i=0;i<(spinorial_calculation+1);i++)
			for(int r=0;r<number_conduction_bands*number_valence_bands*number_k_points_list;r++){
				excitonic_hamiltonian((i+1)*number_conduction_bands*number_valence_bands*number_k_points_list+r,(i+1)*number_conduction_bands*number_valence_bands*number_k_points_list+r).real(real(excitonic_hamiltonian((i+1)*number_conduction_bands*number_valence_bands*number_k_points_list+r,(i+1)*number_conduction_bands*number_valence_bands*number_k_points_list+r))+real(cv_up_down(i,r)));
				excitonic_hamiltonian((i+1)*number_conduction_bands*number_valence_bands*number_k_points_list+r,(i+1)*number_conduction_bands*number_valence_bands*number_k_points_list+r).imag(imag(excitonic_hamiltonian((i+1)*number_conduction_bands*number_valence_bands*number_k_points_list+r,(i+1)*number_conduction_bands*number_valence_bands*number_k_points_list+r))+imag(cv_up_down(i,r)));
			}		
	}
	///cout<<excitonic_hamiltonian<<endl;
	cout<<"Conjugating HBSE"<<endl;
	if(tamn_dancoff==0){
		///#pragma omp parallel for collapse(2)
		for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++)
			for(int j=0;j<spin_dimension_bse_hamiltonian_4_frac_tdf;j++)
				excitonic_hamiltonian(spin_dimension_bse_hamiltonian_4_frac_tdf+i,spin_dimension_bse_hamiltonian_4_frac_tdf+j)=-conj(excitonic_hamiltonian(i,j));
	}
	cout<<"building 00 finished"<<endl;
	//cout<<"hermicity h: "<<accu((excitonic_hamiltonian-conj(excitonic_hamiltonian).t()))<<endl;
	//cout<<excitonic_hamiltonian<<endl;
	///cout<<conj(excitonic_hamiltonian).t()<<endl;
};
void Excitonic_Hamiltonian::add_coupling_part(){
	int offset=spin_dimension_bse_hamiltonian_4_frac_tdf;
	///building entry 01 (coupling part)
	cout<<"building 01"<<endl;
	///building v
	cout<<"building v"<<endl;
	cout<<"beginning extracting rho"<<endl;
	arma::vec zeros_vec(3,arma::fill::zeros);
	arma::cx_mat rho_q_diagk_vc=get<2>(dipole_elements->pull_values(excitonic_momentum,-excitonic_momentum,zeros_vec,1,0,0,1,1,0,threshold_proximity));
	cout<<"ending extracting rho"<<endl;
	arma::cx_vec zeros_long_vec((spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list,arma::fill::zeros);
	arma::cx_mat temporary_matrix1((spinorial_calculation+1)*number_conduction_bands*number_valence_bands*number_k_points_list,number_g_points_list);
	arma::cx_double factor_v;
	factor_v.real((2-spinorial_calculation)/(number_k_points_list*cell_volume));
	factor_v.imag(0.0);

	for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++)
		for(int r=0;r<number_g_points_list;r++){
			temporary_matrix1(i,r).real(real(factor_v*(v_coulomb_g(r)*conj(rho_q_diagk_vc(i,r)))));
			temporary_matrix1(i,r).imag(imag(factor_v*(v_coulomb_g(r)*conj(rho_q_diagk_vc(i,r)))));
		}
	rho_q_diagk_cv.submat(spin_dimension_bse_hamiltonian_2,0,spin_dimension_bse_hamiltonian_4-1,number_g_points_list-1)=rho_q_diagk_vc;
	rho_q_diagk_vc.reset();

	cout<<"second part"<<endl;
	////excluding G=0 maybe not needed
	int g_point_0=int(number_g_points_list/2);
	temporary_matrix1.col(g_point_0)=zeros_long_vec;
	
	arma::cx_mat rho_q_diagk_cv_tmp(spin_dimension_bse_hamiltonian_2,number_g_points_list);
	rho_q_diagk_cv_tmp=rho_q_diagk_cv.submat(0,0,spin_dimension_bse_hamiltonian_2-1,number_g_points_list-1);
	rho_q_diagk_cv_tmp.col(g_point_0)=zeros_long_vec;
	
	for(int i=0;i<(spinorial_calculation+1);i++)
		for(int j=0;j<(spinorial_calculation+1);j++)
			for(int r=0;r<dimension_bse_hamiltonian;r++)
				for(int s=0;s<dimension_bse_hamiltonian;s++)
					for(int g=0;g<number_g_points_list;g++){
						excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,offset+j*3*number_conduction_bands*number_valence_bands*number_k_points_list+s).real(excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,offset+j*3*number_conduction_bands*number_valence_bands*number_k_points_list+s).real()+real(temporary_matrix1(j*number_conduction_bands*number_valence_bands*number_k_points_list+s,g)*(rho_q_diagk_cv_tmp(i*number_conduction_bands*number_valence_bands*number_k_points_list+r,g))));
						excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,offset+j*3*number_conduction_bands*number_valence_bands*number_k_points_list+s).imag(excitonic_hamiltonian(i*3*number_conduction_bands*number_valence_bands*number_k_points_list+r,offset+j*3*number_conduction_bands*number_valence_bands*number_k_points_list+s).imag()+imag(temporary_matrix1(j*number_conduction_bands*number_valence_bands*number_k_points_list+s,g)*(rho_q_diagk_cv_tmp(i*number_conduction_bands*number_valence_bands*number_k_points_list+r,g))));
					}

	temporary_matrix1.reset();
	zeros_long_vec.reset();
	rho_q_diagk_cv_tmp.reset();

	///building W
	///cout<<excitonic_hamiltonian<<endl;
	cout<<"building w"<<endl;
	arma::cx_double factor_w;
	factor_w.real(-1.0/(number_k_points_list*cell_volume));
	factor_w.imag(0.0);
	///double factor_w=-1;

	cout<<"beginning extracting rho"<<endl;
	arma::cx_mat rho_q_kk_cv=get<2>(dipole_elements->pull_values(-excitonic_momentum,zeros_vec,excitonic_momentum,0,0,1,0,0,0,threshold_proximity));
	arma::cx_mat rho_q_kk_vc=get<2>(dipole_elements->pull_values(-excitonic_momentum,-excitonic_momentum,zeros_vec,0,0,0,1,0,0,threshold_proximity));
	cout<<"ending extracting rho"<<endl;

	arma::cx_mat temporary_matrix3(number_k_points_list,number_g_points_list);
	arma::cx_vec temporary_vector3(number_conduction_bands*number_valence_bands*number_k_points_list);
	int spin2; int spin1; int spinv1; int spinc1;
	
	for(int spin1=0;spin1<(3*spinorial_calculation+1);spin1++){
		spinv1=exciton_spin(0,spin1);
		spinc1=exciton_spin(1,spin1);
		if((spin1==0)||(spin1==3*spinorial_calculation))
			spin2=spin1;
		else
			spin2=(2-spin1)+1;
		for(int c1=0;c1<number_conduction_bands;c1++)
			for(int v1=0;v1<number_valence_bands;v1++)
				for(int k1=0;k1<number_k_points_list;k1++){
					for(int c2=0;c2<number_conduction_bands;c2++)
						for(int k2=0;k2<number_k_points_list;k2++){
							for(int s=0;s<number_g_points_list;s++){
								temporary_matrix3(k2,s).real(0.0);
								temporary_matrix3(k2,s).imag(0.0);
								for(int g=0;g<number_g_points_list;g++){
									temporary_matrix3(k2,s).real(temporary_matrix3(k2,s).real()+real((rho_q_kk_cv(spinv1*number_valence_bands*number_conduction_bands*number_k_points_list*number_k_points_list+c2*number_valence_bands*number_k_points_list*number_k_points_list+v1*number_k_points_list*number_k_points_list+k2*number_k_points_list+k1,g))*v_coulomb_gg(g,s,k2*number_k_points_list+k1)));
									temporary_matrix3(k2,s).imag(temporary_matrix3(k2,s).imag()+imag((rho_q_kk_cv(spinv1*number_valence_bands*number_conduction_bands*number_k_points_list*number_k_points_list+c2*number_valence_bands*number_k_points_list*number_k_points_list+v1*number_k_points_list*number_k_points_list+k2*number_k_points_list+k1,g))*v_coulomb_gg(g,s,k2*number_k_points_list+k1)));
								}
							}
							for(int v2=0;v2<number_valence_bands;v2++){
								temporary_vector3(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).real(0.0);
								temporary_vector3(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).imag(0.0);
								for(int g=0;g<number_g_points_list;g++){
									temporary_vector3(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).real(temporary_vector3(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).real()+real(factor_w*temporary_matrix3(k2,g)*conj(rho_q_kk_vc(spinc1*number_valence_bands*number_conduction_bands*number_k_points_list*number_k_points_list+v2*number_conduction_bands*number_k_points_list*number_k_points_list+c1*number_k_points_list*number_k_points_list+k2*number_k_points_list+k1,g))));
									temporary_vector3(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).imag(temporary_vector3(c2*number_valence_bands*number_k_points_list+v2*number_k_points_list+k2).imag()+imag(factor_w*temporary_matrix3(k2,g)*conj(rho_q_kk_vc(spinc1*number_valence_bands*number_conduction_bands*number_k_points_list*number_k_points_list+v2*number_conduction_bands*number_k_points_list*number_k_points_list+c1*number_k_points_list*number_k_points_list+k2*number_k_points_list+k1,g))));
								}
							}
						}
						excitonic_hamiltonian.submat(spin1*number_conduction_bands*number_valence_bands*number_k_points_list+c1*number_valence_bands*number_k_points_list+v1*number_k_points_list+k1,
						offset+spin2*number_conduction_bands*number_valence_bands*number_k_points_list,spin1*number_conduction_bands*number_valence_bands*number_k_points_list+c1*number_valence_bands*number_k_points_list+v1*number_k_points_list+k1,
						offset+(spin2+1)*number_conduction_bands*number_valence_bands*number_k_points_list-1)=temporary_vector3.t();
				}
	}
	temporary_matrix3.reset();
	temporary_vector3.reset();
	rho_q_kk_cv.reset();
	rho_q_kk_vc.reset();

	//cout<<excitonic_hamiltonian<<endl;
	cout<<"Conjugating HBSE"<<endl;
	//#pragma omp parallel for collapse(2)	
	for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++)
		for(int j=0;j<spin_dimension_bse_hamiltonian_4_frac_tdf;j++)
			excitonic_hamiltonian(offset+i,j)=-conj(excitonic_hamiltonian(j,offset+i));
	cout<<"building 01 finished"<<endl;
	///cout<<excitonic_hamiltonian<<endl;

};
std::tuple<arma::cx_mat,arma::cx_mat> Excitonic_Hamiltonian::extract_hbse_and_rcv(arma::vec excitonic_momentum_tmp,double eta,Coulomb_Potential *coulomb_potential,Dielectric_Function *dielectric_function,int adding_screening,int tamn_dancoff,int order_approximation,int number_integration_points,int reading_W,int ipa){
	pull_coulomb_potentials(coulomb_potential,dielectric_function,adding_screening,excitonic_momentum_tmp,eta,order_approximation,number_integration_points,reading_W,0);
	pull_resonant_part_and_rcv(excitonic_momentum_tmp,ipa);
	if(tamn_dancoff==0)
		add_coupling_part();
	return{excitonic_hamiltonian,rho_q_diagk_cv};
};
void Excitonic_Hamiltonian::spin_transformation(){
	int dimension=4;
	arma::cx_mat U(dimension,dimension,arma::fill::zeros);
	
	///U(s1s2,S) (S(M))=11 10 00 1-1
	U(0,0).real(1.0);
	U(1,1).real(1/sqrt(2));
	U(2,1).real(1/sqrt(2));
	U(1,2).real(1/sqrt(2));
	U(2,2).real(-1/sqrt(2));
	U(3,3).real(1.0);
	
	//for(int i=0;i<dimension;i++){
	//	for(int j=0;j<dimension;j++){
	//		cout<<U(i,j)<<"  ";
	//	}
	//	cout<<endl;
	//}
	arma::cx_double variable_tmp;
	#pragma omp parallel for collapse(4) private(variable_tmp) shared(excitonic_hamiltonian)
	for(int i=0;i<dimension;i++)
		for(int j=0;j<dimension;j++)
			for(int r=0;r<dimension_bse_hamiltonian;r++)
				for(int s=0;s<dimension_bse_hamiltonian;s++){
					variable_tmp.real(0.0); variable_tmp.imag(0.0);
					for(int k=0;k<dimension;k++)
						for(int l=0;l<dimension;l++)
							variable_tmp+=U(i,l)*excitonic_hamiltonian(l*dimension_bse_hamiltonian+r,k*dimension_bse_hamiltonian+s)*U(k,j);
					excitonic_hamiltonian(i*dimension_bse_hamiltonian+r,j*dimension_bse_hamiltonian+s)=variable_tmp;
				}
	if(tamn_dancoff==0){
		#pragma omp parallel for collapse(4) private(variable_tmp) shared(excitonic_hamiltonian)
		for(int i=0;i<dimension;i++)
			for(int j=0;j<dimension;j++)
				for(int r=0;r<dimension_bse_hamiltonian;r++)
					for(int s=0;s<dimension_bse_hamiltonian;s++){
						variable_tmp.real(0.0); variable_tmp.imag(0.0);
						for(int k=0;k<dimension;k++)
							for(int l=0;l<dimension;l++)
								variable_tmp+=U(i,l)*excitonic_hamiltonian(l*dimension_bse_hamiltonian+r,spin_dimension_bse_hamiltonian_4+k*dimension_bse_hamiltonian+s)*U(k,j);
						excitonic_hamiltonian(i*dimension_bse_hamiltonian+r,spin_dimension_bse_hamiltonian_4+j*dimension_bse_hamiltonian+s)=variable_tmp;
					}
		#pragma omp parallel for collapse(2)
		for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++)
			for(int j=0;j<spin_dimension_bse_hamiltonian_4_frac_tdf;j++)
				excitonic_hamiltonian(spin_dimension_bse_hamiltonian_4+i,j)=-conj(excitonic_hamiltonian(j,spin_dimension_bse_hamiltonian_4+i));
		#pragma omp parallel for collapse(2)
		for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++)
			for(int j=0;j<spin_dimension_bse_hamiltonian_4_frac_tdf;j++)
				excitonic_hamiltonian(spin_dimension_bse_hamiltonian_4_frac_tdf+i,spin_dimension_bse_hamiltonian_4_frac_tdf+j)=-conj(excitonic_hamiltonian(i,j));
	}
};

/// usual diagonalization routine
std::tuple<arma::cx_vec,arma::cx_mat> Excitonic_Hamiltonian::common_diagonalization(int ipa){
	int dimension=(2-tamn_dancoff)*2;
	cout<<"diagonalization HBSE"<<endl;
	
	///diagonalizing the BSE matrix
	///M_{(bz_number_k_points_list x number_valence_bands x number_conduction_bands)x(bz_number_k_points_list x number_valence_bands x number_conduction_bands)}
	if (spinorial_calculation == 1){
		/////separating the excitonic hamiltonian in two blocks; the ones associated to the magnons and excitons
		//cout<<"BEFORE SPIN TRANSFORMATION"<<endl;
		//for(int i=0;i<spin_dimension_bse_hamiltonian_4_mult_tdf;i++){
		//	for(int j=0;j<spin_dimension_bse_hamiltonian_4_mult_tdf;j++){
		//		cout<<excitonic_hamiltonian(i,j)<<"  ";
		//	}
		//	cout<<endl;
		//}
		//
		spin_transformation();
		arma::cx_mat excitonic_hamiltonian_0(spin_dimension_bse_hamiltonian_4_frac_tdf,spin_dimension_bse_hamiltonian_4_frac_tdf);
		arma::cx_mat excitonic_hamiltonian_1(spin_dimension_bse_hamiltonian_4_frac_tdf,spin_dimension_bse_hamiltonian_4_frac_tdf);
		arma::cx_vec exc_eigenvalues(spin_dimension_bse_hamiltonian_4_mult_tdf); 
		arma::cx_mat exc_eigenvectors(spin_dimension_bse_hamiltonian_4_mult_tdf,spin_dimension_bse_hamiltonian_4_mult_tdf);
		

		for(int i=0;i<2;i++)
			for(int j=0;j<2;j++){
				excitonic_hamiltonian_0.submat(i*dimension_bse_hamiltonian,j*dimension_bse_hamiltonian,(i+1)*dimension_bse_hamiltonian-1,(j+1)*dimension_bse_hamiltonian-1)=
					excitonic_hamiltonian.submat((i+1)*dimension_bse_hamiltonian,(j+1)*dimension_bse_hamiltonian,(i+2)*dimension_bse_hamiltonian-1,(j+2)*dimension_bse_hamiltonian-1);
				excitonic_hamiltonian_1.submat(i*dimension_bse_hamiltonian,j*dimension_bse_hamiltonian,(i+1)*dimension_bse_hamiltonian-1,(j+1)*dimension_bse_hamiltonian-1)=
					excitonic_hamiltonian.submat((i*3)*dimension_bse_hamiltonian,(j*3)*dimension_bse_hamiltonian,(i*3+1)*dimension_bse_hamiltonian-1,(j*3+1)*dimension_bse_hamiltonian-1);
			}
		if(tamn_dancoff==0){
			for(int i=0;i<2;i++)
				for(int j=0;j<2;j++){
					excitonic_hamiltonian_0.submat(i*dimension_bse_hamiltonian,2*dimension_bse_hamiltonian+j*dimension_bse_hamiltonian,(i+1)*dimension_bse_hamiltonian-1,2*dimension_bse_hamiltonian+(j+1)*dimension_bse_hamiltonian-1)=
						excitonic_hamiltonian.submat((i+1)*dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian_4+(j+1)*dimension_bse_hamiltonian,(i+2)*dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian_4+(j+2)*dimension_bse_hamiltonian-1);
					excitonic_hamiltonian_1.submat(i*dimension_bse_hamiltonian,2*dimension_bse_hamiltonian+j*dimension_bse_hamiltonian,(i+1)*dimension_bse_hamiltonian-1,2*dimension_bse_hamiltonian+(j+1)*dimension_bse_hamiltonian-1)=
						excitonic_hamiltonian.submat((i*3)*dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian_4+(j*3)*dimension_bse_hamiltonian,(i*3+1)*dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian_4+(j*3+1)*dimension_bse_hamiltonian-1);
				}
			for(int i=0;i<2;i++)
				for(int j=0;j<2;j++){
					excitonic_hamiltonian_0.submat(i*dimension_bse_hamiltonian+2*dimension_bse_hamiltonian,j*dimension_bse_hamiltonian,(i+1)*dimension_bse_hamiltonian-1+2*dimension_bse_hamiltonian,(j+1)*dimension_bse_hamiltonian-1)=
						excitonic_hamiltonian.submat((i+1)*dimension_bse_hamiltonian+spin_dimension_bse_hamiltonian_4,(j+1)*dimension_bse_hamiltonian,(i+2)*dimension_bse_hamiltonian-1+spin_dimension_bse_hamiltonian_4,(j+2)*dimension_bse_hamiltonian-1);
					excitonic_hamiltonian_1.submat(i*dimension_bse_hamiltonian+2*dimension_bse_hamiltonian,j*dimension_bse_hamiltonian,(i+1)*dimension_bse_hamiltonian-1+2*dimension_bse_hamiltonian,(j+1)*dimension_bse_hamiltonian-1)=
						excitonic_hamiltonian.submat((i*3)*dimension_bse_hamiltonian+spin_dimension_bse_hamiltonian_4,(j*3)*dimension_bse_hamiltonian,(i*3+1)*dimension_bse_hamiltonian-1+spin_dimension_bse_hamiltonian_4,(j*3+1)*dimension_bse_hamiltonian-1);
				}
			for(int i=0;i<2;i++)
				for(int j=0;j<2;j++){
					excitonic_hamiltonian_0.submat(i*dimension_bse_hamiltonian+2*dimension_bse_hamiltonian,j*dimension_bse_hamiltonian+2*dimension_bse_hamiltonian,(i+1)*dimension_bse_hamiltonian-1+2*dimension_bse_hamiltonian,(j+1)*dimension_bse_hamiltonian-1+2*dimension_bse_hamiltonian)=
						excitonic_hamiltonian.submat((i+1)*dimension_bse_hamiltonian+spin_dimension_bse_hamiltonian_4,(j+1)*dimension_bse_hamiltonian+spin_dimension_bse_hamiltonian_4,(i+2)*dimension_bse_hamiltonian-1+spin_dimension_bse_hamiltonian_4,(j+2)*dimension_bse_hamiltonian-1+spin_dimension_bse_hamiltonian_4);
					excitonic_hamiltonian_1.submat(i*dimension_bse_hamiltonian+2*dimension_bse_hamiltonian,j*dimension_bse_hamiltonian+2*dimension_bse_hamiltonian,(i+1)*dimension_bse_hamiltonian-1+2*dimension_bse_hamiltonian,(j+1)*dimension_bse_hamiltonian-1+2*dimension_bse_hamiltonian)=
						excitonic_hamiltonian.submat((i*3)*dimension_bse_hamiltonian+spin_dimension_bse_hamiltonian_4,(j*3)*dimension_bse_hamiltonian+spin_dimension_bse_hamiltonian_4,(i*3+1)*dimension_bse_hamiltonian-1+spin_dimension_bse_hamiltonian_4,(j*3+1)*dimension_bse_hamiltonian-1+spin_dimension_bse_hamiltonian_4);
				}
		}
		//cout<<"SPIN TRANSFORMED"<<endl;
		//for(int i=0;i<spin_dimension_bse_hamiltonian_4_mult_tdf;i++){
		//	for(int j=0;j<spin_dimension_bse_hamiltonian_4_mult_tdf;j++)
		//		cout<<excitonic_hamiltonian(i,j)<<" ";
		//	cout<<endl;
		//}
		//cout<<"HAMILTONIAN 0"<<endl;
		//cout<<excitonic_hamiltonian_0<<endl;
		//cout<<"HAMILTONIAN 1"<<endl;
		///cout<<excitonic_hamiltonian_1<<endl;

		if(ipa==0){
			arma::cx_vec eigenvalues_0;
			arma::cx_mat eigenvectors_0;
			arma::cx_mat eigenvectors_1;
			arma::cx_vec eigenvalues_1;
			///diagonalizing the two spin channels separately: M=0 and M=\pm1
			if(tamn_dancoff==0)
				arma::eig_gen(eigenvalues_0,eigenvectors_0,excitonic_hamiltonian_0);
			else{
				eigenvalues_0.zeros(spin_dimension_bse_hamiltonian_4_frac_tdf);
				arma::vec eigenvalues_tmp;
				arma::eig_sym(eigenvalues_tmp,eigenvectors_0,excitonic_hamiltonian_0);
				for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++)
					eigenvalues_0(i).real(eigenvalues_tmp(i));
			}
			if(tamn_dancoff==0)
				arma::eig_gen(eigenvalues_1,eigenvectors_1,excitonic_hamiltonian_1);
			else{
				eigenvalues_1.zeros(spin_dimension_bse_hamiltonian_4_frac_tdf);
				arma::vec eigenvalues_tmp;
				eig_sym(eigenvalues_tmp,eigenvectors_1,excitonic_hamiltonian_1);
				for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++)
					eigenvalues_1(i).real(eigenvalues_tmp(i));
			}
			/////ordering the eigenvalues and saving them in a single matrix exc_eigenvalues
			/// normalizing and ordering eigenvectors: saving them in a single matrix exc_eigenvectors
			arma::uvec ordering_0=arma::sort_index(real(eigenvalues_0));
			for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++){
				for(int s=0;s<spin_dimension_bse_hamiltonian_4_frac_tdf;s++)
					exc_eigenvectors(s,i)=eigenvectors_0(s,ordering_0(i)); 
				exc_eigenvalues(i) = eigenvalues_0(ordering_0(i));
			//cout<<exc_eigenvalues(i)<<endl;
			}
			/// separating magnons and excitons
			///to add routine separating the two parts
			arma::uvec ordering_1=arma::sort_index(real(eigenvalues_1));
			for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++){
				for(int s=0;s<spin_dimension_bse_hamiltonian_4_frac_tdf;s++)
					exc_eigenvectors(s+spin_dimension_bse_hamiltonian_4_frac_tdf,i+spin_dimension_bse_hamiltonian_4_frac_tdf)=eigenvectors_1(s,ordering_1(i));
				exc_eigenvalues(i+spin_dimension_bse_hamiltonian_4_frac_tdf) = eigenvalues_1(ordering_1(i));
			}

			for(int i=0;i<spin_dimension_bse_hamiltonian_4_mult_tdf;i++)
				exc_eigenvectors.col(i)=exc_eigenvectors.col(i)/norm(exc_eigenvectors.col(i),2);
		}else{
			arma::cx_mat eigenvectors_1(spin_dimension_bse_hamiltonian_4_frac_tdf,spin_dimension_bse_hamiltonian_4_frac_tdf);
			arma::cx_vec eigenvalues_1(spin_dimension_bse_hamiltonian_4_frac_tdf);
			arma::cx_mat eigenvectors_0(spin_dimension_bse_hamiltonian_4_frac_tdf,spin_dimension_bse_hamiltonian_4_frac_tdf);
			arma::cx_vec eigenvalues_0(spin_dimension_bse_hamiltonian_4_frac_tdf);
			for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++){
				eigenvalues_0(i)=excitonic_hamiltonian_0(i,i);
				eigenvectors_0(i,i).real(1.0); 
				eigenvalues_1(i)=excitonic_hamiltonian_1(i,i);
				eigenvectors_1(i,i).real(1.0); 
			}
			/////ordering the eigenvalues and saving them in a single matrix exc_eigenvalues
			/// normalizing and ordering eigenvectors: saving them in a single matrix exc_eigenvectors
			arma::uvec ordering_0=arma::sort_index(real(eigenvalues_0));
			for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++){
				for(int s=0;s<spin_dimension_bse_hamiltonian_4_frac_tdf;s++)
					exc_eigenvectors(s,i)=eigenvectors_0(s,ordering_0(i)); 
				exc_eigenvalues(i) = eigenvalues_0(ordering_0(i));
			//cout<<exc_eigenvalues(i)<<endl;
			}
			/// separating magnons and excitons
			///to add routine separating the two parts
			arma::uvec ordering_1=arma::sort_index(real(eigenvalues_1));
			for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++){
				for(int s=0;s<spin_dimension_bse_hamiltonian_4_frac_tdf;s++)
					exc_eigenvectors(s+spin_dimension_bse_hamiltonian_4_frac_tdf,i+spin_dimension_bse_hamiltonian_4_frac_tdf)=eigenvectors_1(s,ordering_1(i));
				exc_eigenvalues(i+spin_dimension_bse_hamiltonian_4_frac_tdf) = eigenvalues_1(ordering_1(i));
			}
			for(int i=0;i<spin_dimension_bse_hamiltonian_4_mult_tdf;i++)
				exc_eigenvectors.col(i)=exc_eigenvectors.col(i)/norm(exc_eigenvectors.col(i),2);
		}
		//cx_vec eigenvalues_1(spin_dimension_bse_hamiltonian_tdf); 
		//cx_mat eigenvectors_1(spin_dimension_bse_hamiltonian_tdf,spin_dimension_bse_hamiltonian_tdf);
		//cx_vec eigenvalues_0(spin_dimension_bse_hamiltonian_tdf); 
		//cx_mat eigenvectors_0(spin_dimension_bse_hamiltonian_tdf,spin_dimension_bse_hamiltonian_tdf);
		//lapack_complex_double *temporary_0;
		//temporary_0=(lapack_complex_double*)malloc(spin_dimension_bse_hamiltonian_tdf*spin_dimension_bse_hamiltonian_tdf*sizeof(lapack_complex_double)); 
		//lapack_complex_double *temporary_1;
		//temporary_1=(lapack_complex_double*)malloc(spin_dimension_bse_hamiltonian_tdf*spin_dimension_bse_hamiltonian_tdf*sizeof(lapack_complex_double)); 

		////#pragma omp parallel for collapse(2)
		//for(int i=0;i<spin_dimension_bse_hamiltonian_tdf;i++)
		//	for(int j=0;j<spin_dimension_bse_hamiltonian_tdf;j++){
		//		temporary_0[i*spin_dimension_bse_hamiltonian_tdf+j]=real(excitonic_hamiltonian_0(i,j))+_Complex_I*imag(excitonic_hamiltonian_0(i,j));
		//		temporary_1[i*spin_dimension_bse_hamiltonian_tdf+j]=real(excitonic_hamiltonian_1(i,j))+_Complex_I*imag(excitonic_hamiltonian_1(i,j));
		//	}
	
		//int N=spin_dimension_bse_hamiltonian_tdf;
		//int LDA=spin_dimension_bse_hamiltonian_tdf;
		//int LDVL=1;
		//int LDVR=spin_dimension_bse_hamiltonian_tdf;
		//char JOBVR='V';
		//char JOBVL='N';
		//int matrix_layout = 101;
		//int INFO0; int INFO1;
		//lapack_complex_double *empty;

		//lapack_complex_double *w_0;
		//w_0=(lapack_complex_double*)malloc(N*sizeof(lapack_complex_double));
		//lapack_complex_double *u_0;
		//u_0=(lapack_complex_double*)malloc(N*LDVR*sizeof(lapack_complex_double));
		//lapack_complex_double *w_1;
		//w_1=(lapack_complex_double*)malloc(N*sizeof(lapack_complex_double));
		//lapack_complex_double *u_1;
		//u_1=(lapack_complex_double*)malloc(N*LDVR*sizeof(lapack_complex_double));
		//
		//INFO0 = LAPACKE_zgeev(matrix_layout,JOBVL,JOBVR,N,temporary_0,LDA,w_0,empty,LDVL,u_0,LDVR);
		//INFO1 = LAPACKE_zgeev(matrix_layout,JOBVL,JOBVR,N,temporary_1,LDA,w_1,empty,LDVL,u_1,LDVR);
		//
		//free(temporary_0); free(temporary_1);
		////eig_gen(eigenvalues_1,eigenvectors_1,excitonic_hamiltonian_1);
		////eig_gen(eigenvalues_0,eigenvectors_0,excitonic_hamiltonian_0);
		//for(int i=0;i<spin_dimension_bse_hamiltonian_tdf;i++){
		//	eigenvalues_0(i).real(lapack_complex_double_real(w_0[i]));
		//	eigenvalues_0(i).imag(lapack_complex_double_imag(w_0[i]));
		//	eigenvalues_1(i).real(lapack_complex_double_real(w_1[i]));
		//	eigenvalues_1(i).imag(lapack_complex_double_imag(w_1[i]));
		//}

		
		///free(w_0); free(w_1); free(u_0); free(u_1);
		//cout<<exc_eigenvalues<<endl;
		ofstream transitions_tmp_file;
		transitions_tmp_file.open("tmp_transitions.txt");
		for(int g=0;g<spin_dimension_bse_hamiltonian_4_frac_tdf;g++){
			transitions_tmp_file<<0<<exc_eigenvalues(g)<<endl;
			transitions_tmp_file<<1<<exc_eigenvalues(g+spin_dimension_bse_hamiltonian_4_frac_tdf)<<endl;
		}
		transitions_tmp_file.close();
		return {exc_eigenvalues.subvec(0,spin_dimension_bse_hamiltonian_4_frac_tdf-1),exc_eigenvectors.submat(0,0,spin_dimension_bse_hamiltonian_4_frac_tdf-1,spin_dimension_bse_hamiltonian_4_frac_tdf-1)};
	}else{
		arma::cx_vec exc_eigenvalues(spin_dimension_bse_hamiltonian_2_mult_tdf); 
		arma::cx_mat exc_eigenvectors(spin_dimension_bse_hamiltonian_2_mult_tdf,spin_dimension_bse_hamiltonian_2_mult_tdf);
			
		if(ipa==0){
			arma::cx_mat eigenvectors;
			arma::cx_vec eigenvalues;
			if(tamn_dancoff==0)
				eig_gen(eigenvalues,eigenvectors,excitonic_hamiltonian);
			else{
				eigenvalues.zeros(spin_dimension_bse_hamiltonian_2_mult_tdf);
				arma::vec eigenvalues_tmp;
				arma::eig_sym(eigenvalues_tmp,eigenvectors,excitonic_hamiltonian);
				for(int i=0;i<spin_dimension_bse_hamiltonian_2_mult_tdf;i++)
					eigenvalues(i).real(eigenvalues_tmp(i));
			}
			double exc_norm;
			arma::uvec ordering=arma::sort_index(real(eigenvalues));
			for(int i=0;i<spin_dimension_bse_hamiltonian_2_mult_tdf;i++){
				exc_norm=arma::vecnorm(eigenvectors.col(i));
				for(int s=0;s<spin_dimension_bse_hamiltonian_2_mult_tdf;s++)
					if(exc_norm!=0.0)
						exc_eigenvectors(s,i)=eigenvectors(s,ordering(i))/exc_norm;
					else
						exc_eigenvectors(s,i)=eigenvectors(s,ordering(i));
				exc_eigenvalues(i)=eigenvalues(ordering(i));
				cout<<exc_eigenvalues(i)<<endl;
				}
		}else{
			arma::cx_vec eigenvalues(spin_dimension_bse_hamiltonian_2_mult_tdf); 
			arma::cx_mat eigenvectors(spin_dimension_bse_hamiltonian_2_mult_tdf,spin_dimension_bse_hamiltonian_2_mult_tdf,arma::fill::zeros);
			for(int i=0;i<spin_dimension_bse_hamiltonian_2_mult_tdf;i++){
				eigenvalues(i)=excitonic_hamiltonian(i,i);
				eigenvectors(i,i).real(1.0);
			}
			double exc_norm;
			arma::uvec ordering=arma::sort_index(real(eigenvalues));
			for(int i=0;i<spin_dimension_bse_hamiltonian_2_mult_tdf;i++){
				exc_norm=arma::vecnorm(eigenvectors.col(i));
				for(int s=0;s<spin_dimension_bse_hamiltonian_2_mult_tdf;s++)
					if(exc_norm!=0.0)
						exc_eigenvectors(s,i)=eigenvectors(s,ordering(i))/exc_norm;
					else
						exc_eigenvectors(s,i)=eigenvectors(s,ordering(i));
				exc_eigenvalues(i)=eigenvalues(ordering(i));
				///cout<<exc_eigenvalues(i)<<endl;
				}
		}
		return {exc_eigenvalues,exc_eigenvectors};
	}
};

/// Fastest diagonalization routine
///[1] Structure preserving parallel algorithms for solving the Bethe–Salpeter eigenvalue problem Meiyue Shao, Felipe H. da Jornada, Chao Yang, Jack Deslippe, Steven G. Louie
///[2] Beyond the Tamm-Dancoff approximation for ext.ended systems using exact diagonalization Tobias Sander, Emanuelel Maggio, and Georg Kresse
///tuple<vec,cx_mat> Excitonic_Hamiltonian:: cholesky_diagonalization(double eta)
///{
///	/// diagonalizing the BSE matrix M_{(bz_number_k_points_list x number_valence_bands x number_conduction_bands)x(bz_number_k_points_list x number_valence_bands x number_conduction_bands)}
///	int spin_dimension_bse_hamiltonian_2 = 2*spin_dimension_bse_hamiltonian;
///	cx_mat A=excitonic_hamiltonian.submat(0,0,spin_dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian-1);
///	cx_mat B=excitonic_hamiltonian.submat(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian,2*spin_dimension_bse_hamiltonian-1,2*spin_dimension_bse_hamiltonian-1);
///
///	mat M(spin_dimension_bse_hamiltonian_2, spin_dimension_bse_hamiltonian_2);
///	// Fill top-left submatrix
///	M.submat(0, 0, spin_dimension_bse_hamiltonian - 1, spin_dimension_bse_hamiltonian - 1) = real(A + B);
///	// Fill bottom-left submatrix (as conjugate transpose of top-right submatrix)
///	M.submat(spin_dimension_bse_hamiltonian, 0, spin_dimension_bse_hamiltonian_2 - 1, spin_dimension_bse_hamiltonian - 1) = -imag(A + B).t();
///	// Fill top-right submatrix (as conjugate transpose of bottom-left submatrix)
///	M.submat(0, spin_dimension_bse_hamiltonian, spin_dimension_bse_hamiltonian - 1, spin_dimension_bse_hamiltonian_2 - 1) = -imag(A-B);
///	// Fill bottom-right submatrix
///	M.submat(spin_dimension_bse_hamiltonian, spin_dimension_bse_hamiltonian, spin_dimension_bse_hamiltonian_2 - 1, spin_dimension_bse_hamiltonian_2 - 1) = real(A - B);
///	
///	///symmetrizing
///	M+=M.t();
///	M=M/2;
///	cout<<M.is_symmetric()<<endl;
///
///	cout<<"cholesky factorization"<<endl;
///	/// construct W
///	vec diag_one(spin_dimension_bse_hamiltonian,fill::ones);
///	mat J(spin_dimension_bse_hamiltonian_2,spin_dimension_bse_hamiltonian_2,fill::zeros);
///	J.submat(0,spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian_2-1)=diagmat(diag_one);
///	J.submat(spin_dimension_bse_hamiltonian,0,spin_dimension_bse_hamiltonian_2-1,spin_dimension_bse_hamiltonian-1)=-diagmat(diag_one);
///	
///	mat L(spin_dimension_bse_hamiltonian_2, spin_dimension_bse_hamiltonian_2,fill::zeros);
///	//L=chol(M);
///	
///	int N=spin_dimension_bse_hamiltonian_2;
///	int LDA=spin_dimension_bse_hamiltonian_2;
///	int matrix_layout = 101;
///	int INFO;
///	char UPLO = 'U';
///	double *temporary_0; temporary_0 = (double *)malloc(spin_dimension_bse_hamiltonian_2*spin_dimension_bse_hamiltonian_2*sizeof(double));
///	//#pragma omp parallel for collapse(2)
///	for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++)
///		for(int j=0;j<spin_dimension_bse_hamiltonian_2;j++){
///			temporary_0[i*spin_dimension_bse_hamiltonian+j]=M(i,j);
///		}
///
///	INFO=LAPACKE_dpotrf(matrix_layout, UPLO, N, temporary_0, LDA);
///
///	for (int i = 0; i < spin_dimension_bse_hamiltonian_2; i++)
///		for (int j = i; j < spin_dimension_bse_hamiltonian_2; j++){
///			L(i,j)=temporary_0[i * spin_dimension_bse_hamiltonian_2 + j];
///		}
///
///	free(temporary_0);
///
///	mat W(spin_dimension_bse_hamiltonian_2,spin_dimension_bse_hamiltonian_2);
///	W =  L.t() * J;
///	W = W * L;
///	
///	lapack_complex_float *temporary_1; temporary_1 = (lapack_complex_float *)malloc(spin_dimension_bse_hamiltonian_2*spin_dimension_bse_hamiltonian_2*sizeof(lapack_complex_float));
///	
///	//#pragma omp parallel for collapse(2)
///	for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++)
///		for(int j=0;j<spin_dimension_bse_hamiltonian_2;j++){
///			temporary_1[i*spin_dimension_bse_hamiltonian_2+j]=imag(W(i,j))-_Complex_I*real(W(i,j));
///			//cout<<temporary_1[i*spin_dimension_bse_hamiltonian_2+j]<<endl;
///			cout<<W(i,j)<<" ";
///		}
///
///	float *w;
///	char JOBZ = 'V';
///	char JOBU = 'A';
///	char JOBVT = 'A';
///	int F=spin_dimension_bse_hamiltonian_2;
///	int LDU=spin_dimension_bse_hamiltonian_2;
///	int LDVT=spin_dimension_bse_hamiltonian_2;
///	lapack_complex_float *z1;
///	lapack_complex_float *z2;
///	z1=(lapack_complex_float*)malloc(F*F*sizeof(lapack_complex_float));
///	z2=(lapack_complex_float*)malloc(F*F*sizeof(lapack_complex_float));
///	//// saving all the eigenvalues
///	w = (float *)malloc(spin_dimension_bse_hamiltonian_2 * sizeof(float));
///	float *lwork; lwork=(float*)malloc(4*F*sizeof(float));
///
///	cout<<"singular value decomposition"<<endl;
///	INFO=LAPACKE_cgesvd(matrix_layout,JOBU,JOBVT,F,N,temporary_1,LDA,w,z1,LDU,z2,LDVT,lwork);
///	
///	free(temporary_1);
///
///	vec exc_eigenvalues(spin_dimension_bse_hamiltonian);
///	for(int i=0;i<spin_dimension_bse_hamiltonian;i++){
///		exc_eigenvalues(i)=-w[i];
///		cout<<exc_eigenvalues(i)<<endl;
///	}
///	free(w);
///
///	cx_mat exc_eigenvectors(spin_dimension_bse_hamiltonian_2,spin_dimension_bse_hamiltonian_2);
///	cx_mat z_(spin_dimension_bse_hamiltonian_2,spin_dimension_bse_hamiltonian_2);
///	for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++)
///		for(int j=0;j<spin_dimension_bse_hamiltonian_2;j++)
///			z_(i,j)=z1[i*spin_dimension_bse_hamiltonian_2+j];
///	free(z1); free(z2);
///		
///	cx_double ieta;
///	ieta.real(0.0); ieta.imag(eta);
///
///	exc_eigenvectors.set_real(J*L);
///	exc_eigenvectors=exc_eigenvectors*z_;
///
///	exc_eigenvectors.submat(0,0,spin_dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian-1)=exc_eigenvectors.submat(0,0,spin_dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian-1)*inv(diagmat(sqrt(exc_eigenvalues)+ieta));
///	exc_eigenvectors.submat(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian_2-1,spin_dimension_bse_hamiltonian_2-1)=exc_eigenvectors.submat(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian_2-1,spin_dimension_bse_hamiltonian_2-1)*inv(diagmat(sqrt(exc_eigenvalues)+ieta));
///
///	return{exc_eigenvalues,exc_eigenvectors.submat(0,0,spin_dimension_bse_hamiltonian-1,spin_dimension_bse_hamiltonian-1)};
///};

std::tuple<arma::cx_vec,arma::cx_vec> Excitonic_Hamiltonian::pull_excitonic_oscillator_force(arma::cx_mat excitonic_eigenstates,int tamn_dancoff,int ipa){
	arma::cx_vec oscillator_force_l(spin_dimension_bse_hamiltonian_2,arma::fill::zeros);
	///cout<<excitonic_eigenstates<<endl;
	int g_point_0=int(number_g_points_list/2);
	if((tamn_dancoff==1)||(ipa==1)){
		for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++)
			oscillator_force_l(i)=arma::accu(arma::conj(rho_q_diagk_cv.col(g_point_0))%excitonic_eigenstates.col(i+spin_dimension_bse_hamiltonian_2*(1-tamn_dancoff)));
		return {oscillator_force_l,oscillator_force_l};
	}else{
		arma::cx_vec oscillator_force_r(spin_dimension_bse_hamiltonian_2,arma::fill::zeros);
		for(int i=0;i<spin_dimension_bse_hamiltonian_2;i++){
			oscillator_force_l(i)=arma::accu(arma::conj(rho_q_diagk_cv.submat(0,0,spin_dimension_bse_hamiltonian_2-1,g_point_0))%excitonic_eigenstates.submat(0,i,spin_dimension_bse_hamiltonian_2-1,i));
			oscillator_force_r(i)=arma::accu(arma::conj(rho_q_diagk_cv.submat(spin_dimension_bse_hamiltonian_2,0,spin_dimension_bse_hamiltonian_4-1,g_point_0))%excitonic_eigenstates.submat(spin_dimension_bse_hamiltonian_2,i,spin_dimension_bse_hamiltonian_4-1,i));
		}
		return {oscillator_force_l,oscillator_force_r};
	}
};

arma::cx_mat Excitonic_Hamiltonian:: pull_augmentation_matrix(arma::cx_mat exc_eigenstates,int spin_dimension_bse_hamiltonian_4_frac_tdf){
	arma::cx_mat augmentation_matrix(spin_dimension_bse_hamiltonian_4_frac_tdf,spin_dimension_bse_hamiltonian_4_frac_tdf);
	arma::cx_double temporary_variable;
	
	for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++)
		for(int j=i;j<spin_dimension_bse_hamiltonian_4_frac_tdf;j++){
			temporary_variable.real(0.0);
			temporary_variable.imag(0.0);
			for(int r=0;r<spin_dimension_bse_hamiltonian_4_frac_tdf;r++)
				temporary_variable+=conj(exc_eigenstates(r,i))*exc_eigenstates(r,j);
			augmentation_matrix(i,j)=temporary_variable;
			augmentation_matrix(j,i)=temporary_variable;
		}
	cout<<augmentation_matrix<<endl;
	///inverting the matrix
	arma::cx_mat augmentation_matrix_inv(spin_dimension_bse_hamiltonian_4_frac_tdf,spin_dimension_bse_hamiltonian_4_frac_tdf);
	arma::cx_mat identity(spin_dimension_bse_hamiltonian_4_frac_tdf,spin_dimension_bse_hamiltonian_4_frac_tdf,arma::fill::zeros);
	for(int i=0;i<spin_dimension_bse_hamiltonian_4_frac_tdf;i++)
		identity(i,i).real(1.0);

	arma::solve(augmentation_matrix_inv,augmentation_matrix,identity);
	cout<<augmentation_matrix_inv<<endl;
	return augmentation_matrix_inv;
};


void Excitonic_Hamiltonian:: pull_macroscopic_bse_dielectric_function(arma::cx_vec omegas_path,int number_omegas_path,double eta,string file_macroscopic_dielectric_function_bse_name,double lorentzian,int tamn_dancoff,Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,int order_approximation,int number_integration_points,int reading_W,int ipa)
{
	cout << "Calculating dielectric tensor..." << endl;
	double factor=conversion_parameter/(minval*minval*cell_volume*number_k_points_list);
	cout<<cell_volume<<endl;

	arma::cx_cube dielectric_tensor_bse(3,3,number_omegas_path,arma::fill::zeros);
	arma::cx_vec average_dielectric_tensor_bse(number_omegas_path,arma::fill::zeros);
	arma::cx_double ieta;	ieta.real(0.0);	ieta.imag(eta);
	arma::cx_double ilorentzian; ilorentzian.real(0.0); ilorentzian.imag(lorentzian);
	arma::cx_vec temporary_variable(number_omegas_path);
	arma::mat excitonic_momentum(3,3,arma::fill::zeros);
	arma::vec excitonic_momentum1(3,arma::fill::zeros);
	std::tuple<arma::cx_vec,arma::cx_vec> oscillators_forces;

	arma::cx_mat delta(3,3,arma::fill::zeros);
	for(int i=0;i<3;i++)
		delta(i,i).real(1.0);

	arma::cx_vec exc_eigenvalues;
	arma::cx_mat exc_eigenstates;
	arma::cx_vec exc_oscillator_force_l;
	arma::cx_vec exc_oscillator_force_r;

	arma::cx_mat augmentation_matrix_inv;

	ofstream dielectric_tmp_file;
	dielectric_tmp_file.open("tmp_dielectric.txt");
	
	//for(int r=0;r<3;r++)
	//	excitonic_momentum(r,0)=bravais_lattice(r,2)/arma::vecnorm(bravais_lattice.col(2));
	excitonic_momentum(0,0)=minval;
	excitonic_momentum(1,1)=minval;
	excitonic_momentum(2,1)=minval;
	///excitonic_momentum.col(2)=arma::cross(excitonic_momentum.col(1),excitonic_momentum.col(0));
	///for(int r=0;r<3;r++)
	///	excitonic_momentum(r,2)=excitonic_momentum(r,2)/arma::vecnorm(excitonic_momentum.col(2));

	///cout<<"ARE SPECTRA ORTHONORMAL?"<<endl;
	///cout<<arma::dot(excitonic_momentum.col(0),excitonic_momentum.col(1))<<endl;
	///cout<<arma::dot(excitonic_momentum.col(1),excitonic_momentum.col(2))<<endl;
	///cout<<arma::dot(excitonic_momentum.col(2),excitonic_momentum.col(0))<<endl;

	for (int i = 0; i < 1; i++)
		for (int j = 0; j < 1; j++)
			if(i==j){
				///for(int r=0;r<3;r++)
				///	excitonic_momentum1(r)=minval*excitonic_momentum(r,i)/arma::vecnorm(excitonic_momentum.col(i));
				excitonic_momentum1=excitonic_momentum.col(i);
				if(ipa==0)
					pull_coulomb_potentials(coulomb_potential,dielectric_function,adding_screening,excitonic_momentum1,eta,order_approximation,number_integration_points,reading_W,0);
				pull_resonant_part_and_rcv(excitonic_momentum1,ipa);
				///cout<<excitonic_hamiltonian<<endl;
				///cout<<k_points_differences<<endl;
				if((tamn_dancoff==0)&&(ipa==0)){
					pull_coulomb_potentials(coulomb_potential,dielectric_function,adding_screening,excitonic_momentum1,eta,order_approximation,number_integration_points,reading_W,1);
					add_coupling_part();
				}
				////cout<<k_points_differences<<endl;
				///cout<<excitonic_hamiltonian<<endl;
				cout<<"rho_cv"<<endl;
				///cout<<rho_q_diagk_cv<<endl;
				std::tuple<arma::cx_vec,arma::cx_mat> eigenvalues_and_eigenstates = common_diagonalization(ipa);
				exc_eigenvalues=get<0>(eigenvalues_and_eigenstates);
				exc_eigenstates=get<1>(eigenvalues_and_eigenstates);
				if((tamn_dancoff==0)&&(ipa==0))
					augmentation_matrix_inv=pull_augmentation_matrix(exc_eigenstates,spin_dimension_bse_hamiltonian_4_frac_tdf);
				cout<<"exciton eigenvalues"<<endl;			
				///cout<<exc_eigenvalues<<endl;
				///cout<<exc_eigenstates<<endl;
				oscillators_forces=pull_excitonic_oscillator_force(exc_eigenstates,tamn_dancoff,ipa);
				exc_oscillator_force_l=get<0>(oscillators_forces);
				exc_oscillator_force_r=get<1>(oscillators_forces);
				cout<<"oscillator forces"<<endl;
				cout<<rho_q_diagk_cv<<endl;
				///cout<<exc_oscillator_force_l<<" "<<exc_oscillator_force_r<<endl;
				if((tamn_dancoff==0)&&(ipa==0)){
					for(int s=0;s<number_omegas_path;s++){
						temporary_variable(s)=0.0;
						for(int l=0;l<spin_dimension_bse_hamiltonian_2;l++)
							for(int r=0;r<spin_dimension_bse_hamiltonian_2;r++){
								temporary_variable(s)+=conj(exc_oscillator_force_l(l))*exc_oscillator_force_l(r)*augmentation_matrix_inv(l,r)/(omegas_path(s)-exc_eigenvalues(l)+ilorentzian);
								temporary_variable(s)+=conj(exc_oscillator_force_r(l))*exc_oscillator_force_r(r)*augmentation_matrix_inv(l,r)/(omegas_path(s)-exc_eigenvalues(spin_dimension_bse_hamiltonian_2+l)+ilorentzian);
								temporary_variable(s)+=conj(exc_oscillator_force_l(l))*exc_oscillator_force_r(r)*augmentation_matrix_inv(l,r)/(omegas_path(s)-exc_eigenvalues(l)+ilorentzian);
								temporary_variable(s)+=conj(exc_oscillator_force_r(l))*exc_oscillator_force_l(r)*augmentation_matrix_inv(l,r)/(omegas_path(s)-exc_eigenvalues(spin_dimension_bse_hamiltonian_2+l)+ilorentzian);
							}
						dielectric_tensor_bse(i,j,s)=delta(i,j)-factor*temporary_variable(s);
						dielectric_tmp_file<<dielectric_tensor_bse(i,j,s)<<endl;
						average_dielectric_tensor_bse(s)+=dielectric_tensor_bse(i,j,s);
					}
				}else{
					for(int s=0;s<number_omegas_path;s++){
						temporary_variable(s)=0.0;
						for(int m=0;m<spin_dimension_bse_hamiltonian_2;m++){
							temporary_variable(s)+=conj(exc_oscillator_force_l(m))*exc_oscillator_force_r(m)/(omegas_path(s)-exc_eigenvalues(m+spin_dimension_bse_hamiltonian_2*(1-tamn_dancoff))+ilorentzian);
							///cout<<"oscillator strength "<<conj(exc_oscillator_force_l(m))*exc_oscillator_force_r(m)<<endl;
						}
						dielectric_tensor_bse(i,j,s)=delta(i,j)-factor*temporary_variable(s);
						dielectric_tmp_file<<dielectric_tensor_bse(i,j,s)<<endl;
						average_dielectric_tensor_bse(s)+=dielectric_tensor_bse(i,j,s);
					}
				}
				
				exc_eigenvalues.reset();
				exc_eigenstates.reset();
				(get<0>(eigenvalues_and_eigenstates)).reset();
				(get<1>(eigenvalues_and_eigenstates)).reset();
			}
	dielectric_tmp_file.close();

	cout<<"Printing over file"<<endl;
	ofstream dielectric_tensor_file;
	dielectric_tensor_file.open(file_macroscopic_dielectric_function_bse_name);
	///writing the dielectric function (in the optical limit) in a file
	dielectric_tensor_file<<"### omega xx xy xz yx yy yz zx zy zz"<<endl;
	for(int s=0;s<number_omegas_path;s++){
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				if(i==j)
					dielectric_tensor_file<<omegas_path(s)<<" "<<dielectric_tensor_bse(i,j,s)<<" ";
		dielectric_tensor_file<<average_dielectric_tensor_bse(s).imag()/3<<endl;
	}

	dielectric_tensor_file.close();
};

void Excitonic_Hamiltonian::print(arma::vec excitonic_momentum_tmp,double eta,int tamn_dancoff,Coulomb_Potential* coulomb_potential,Dielectric_Function* dielectric_function,int adding_screening,int order_approximation,int number_integration_points,int reading_W,int ipa){
	
	pull_coulomb_potentials(coulomb_potential,dielectric_function,adding_screening,excitonic_momentum,eta,order_approximation,number_integration_points,reading_W,0);	
	pull_resonant_part_and_rcv(excitonic_momentum_tmp,ipa);
	if(tamn_dancoff==0)
		add_coupling_part();
	cout<<number_k_points_list<<endl;
	
	cout<<"BSE hamiltoian..."<<endl;
	for(int i=0;i<spin_dimension_bse_hamiltonian_4_mult_tdf;i++){
		for(int j=0;j<spin_dimension_bse_hamiltonian_4_mult_tdf;j++)
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

int main(int argc, char** argv){

	cout<<minval<<" "<<conversion_parameter<<endl;
	//if(my_rank==0){
	double fermi_energy = 15.5124;
	///double fermi_energy=6.800;
	////Initializing Lattice
	string file_crystal_bravais_name="bravais.lattice.data";
	string file_crystal_coordinates_name="atoms.data";
	//string file_crystal_bravais_name="bravais.lattice_si.data";
	//string file_crystal_coordinates_name="atoms_si.data";
	int number_atoms=4;
	Crystal_Lattice crystal(file_crystal_bravais_name,file_crystal_coordinates_name,number_atoms);
	double volume=crystal.pull_volume();
	arma::mat bravais_lattice=crystal.pull_bravais_lattice();
	cout<<bravais_lattice<<endl;
	cout<<bravais_lattice.col(0)<<endl;
	cout<<volume<<endl;
	arma::mat atoms_coordinates=crystal.pull_atoms_coordinates();
	crystal.print();

	////Initializing k points list
	arma::vec shift; shift.zeros(3);
	////shift in crystal coordinates
	shift(0)=0.000;
	shift(1)=0.000;
	shift(1)=0.000;
	string file_k_points_name="k_points_list_si.dat";
	int number_k_points_list=200;
	int crystal_coordinates=1;
	int random_generator=1;
	K_points k_points(&crystal,shift,number_k_points_list);
	k_points.push_k_points_list_values(file_k_points_name,crystal_coordinates,random_generator);
	arma::mat k_points_list=k_points.pull_k_points_list_values();
	k_points.print();
	arma::mat primitive_vectors=k_points.pull_primitive_vectors();

	//////Initializing g points list
	arma::vec shift_g; shift_g.zeros(3);
	//double cutoff_g_points_list=2;
	double cutoff_g_points_list=1;
	int dimension_g_points_list=3;
	arma::vec direction_cutting(3); direction_cutting(0)=1; direction_cutting(1)=1; direction_cutting(2)=1;
	G_points g_points(&crystal,cutoff_g_points_list,dimension_g_points_list,direction_cutting,shift_g);
	arma::mat g_points_list=g_points.pull_g_points_list_values(); 
	int number_g_points_list=g_points.pull_number_g_points_list();
	cout<<"G points: "<<number_g_points_list<<endl;
	int g_point0=int(number_g_points_list/2);
	cout<<g_points_list.col(g_point0)<<endl;
	g_points.print();

	//////Initializing Coulomb potential
	double minimum_k_point_modulus=0.0;
	int dimension_potential=3;
	double radius=0.0;
	Coulomb_Potential coulomb_potential(&k_points,&g_points,minimum_k_point_modulus,dimension_potential,direction_cutting,volume,radius);
	string file_coulomb_name="coulomb.dat"; 
	//int number_k_points_c=10000; double max_k_points_radius_c=6.0;
	//int direction_profile_xyz=0;
	//coulomb_potential.print_profile(number_k_points_c,max_k_points_radius_c,file_coulomb_name,direction_profile_xyz);

	////Initializing the Tight Binding hamiltonian (saving the Wannier functions centers)
	ifstream file_htb; ifstream file_centers; string seedname;
	string wannier90_hr_file_name="nio_hr.dat";
	string wannier90_centers_file_name="nio_centres.xyz";
	///string wannier90_hr_file_name="silicon_hr.dat_8bands_10_20";
	///string wannier90_centers_file_name="silicon_centres.xyz_8bands_10_20";
	
	bool dynamic_shifting=false;
	int spinorial_calculation = 1;
	double little_shift=0.00;
	double scissor_operator=1.00;
	///double scissor_operator=3.00;
	int number_primitive_cells=597;
	int number_wannier_functions=16;
	Hamiltonian_TB htb(wannier90_hr_file_name,wannier90_centers_file_name,fermi_energy,spinorial_calculation,number_atoms,dynamic_shifting,little_shift,scissor_operator,bravais_lattice,number_primitive_cells,number_wannier_functions);

	/// 0 no spinors, 1 collinear spinors, 2 non-collinear spinors (implementing 0 and 1 cases)
	int number_wannier_centers=htb.pull_number_wannier_functions();
	int htb_basis_dimension=htb.pull_htb_basis_dimension();
	arma::vec k_point; k_point.zeros(3); k_point(0)=minval;
	///BANDS STRUCTURE
	//string bands_file_name="bands.data";
	//string k_points_bands_file_name="k_points_bands.data";
	//int number_k_points_bands=3;
	//htb.pull_bands(bands_file_name,k_points_bands_file_name,number_k_points_bands,14,2,crystal_coordinates,primitive_vectors);

	//////Initializing dipole elements
	int number_conduction_bands_selected_diel=2;
	int number_valence_bands_selected_diel=6;
	int number_conduction_bands_selected=2;
	int number_valence_bands_selected=6;


	////Initializing Real Space Wannier functions
	arma::vec number_points_real_space_grid(3);
	arma::vec number_unit_cells_supercell(3);
	string seedname_files_xsf="Calc_";
	number_unit_cells_supercell(0)=3;
	number_unit_cells_supercell(1)=3;
	number_unit_cells_supercell(2)=3;
	number_points_real_space_grid(0)=162;
	number_points_real_space_grid(1)=162;
	number_points_real_space_grid(2)=162;
	
	Real_space_wannier real_space_wannier(number_points_real_space_grid,number_unit_cells_supercell,spinorial_calculation,seedname_files_xsf,number_wannier_functions,volume,number_atoms,atoms_coordinates);
	arma::vec which_cell(3);
	which_cell(0)=0;
	which_cell(1)=0;
	which_cell(2)=0;
	double isovalue_pos=20;
	double isovalue_neg=-10;
	//string wannier_file_name="test.xsf";
	//real_space_wannier.print(1,which_cell,0,isovalue_pos,isovalue_neg,wannier_file_name);

	arma::vec number_primitive_cells_integration(3);
	number_primitive_cells_integration(0)=1;
	number_primitive_cells_integration(1)=1;
	number_primitive_cells_integration(2)=1;
	
	Dipole_Elements dipole_elements(number_k_points_list,k_points_list,number_g_points_list,g_points_list,number_wannier_centers,number_valence_bands_selected,number_conduction_bands_selected,&htb,spinorial_calculation,&real_space_wannier,number_primitive_cells_integration,number_unit_cells_supercell,number_points_real_space_grid);
	arma::vec zeros(3,arma::fill::zeros);
	arma::vec excitonic_momentum(3,arma::fill::zeros);
	excitonic_momentum(0)=minval;
	///dipole_elements.print(excitonic_momentum,zeros,zeros,1,0,0,0,0.0);
	/////////Initializing dielectric function
	Dielectric_Function dielectric_function(&dipole_elements,number_k_points_list,number_g_points_list,g_points_list,number_valence_bands_selected_diel,number_conduction_bands_selected_diel,&coulomb_potential,spinorial_calculation,volume);
	arma::cx_double omega; omega.real(0.0); omega.imag(0.0); 
	double eta=0.0; 
	int order_approximation=0;
	int number_omegas_path=3200;
	arma::cx_vec omegas_path(number_omegas_path);
	arma::cx_double max_omega=10.00;
	arma::cx_double min_omega=0.00;
	arma::cx_vec macroscopic_dielectric_function(number_omegas_path);
	for(int i=0;i<number_omegas_path;i++)
		omegas_path(i)=(min_omega+double(i)/double(number_omegas_path)*(max_omega-min_omega));
	
	int adding_screening=0;
	double lorentzian=0.05;
	int tamn_dancoff=1;
	int ipa=0;
	int insulator_metal=0;
	double threshold_proximity=0.2;
	arma::mat k_points_differences=k_points.pull_k_point_differences();
	Excitonic_Hamiltonian htbse(number_valence_bands_selected,number_conduction_bands_selected,k_points_list,number_k_points_list,g_points_list,number_g_points_list,spinorial_calculation,htb_basis_dimension,&dipole_elements,volume,tamn_dancoff,insulator_metal,k_points_differences,threshold_proximity);
	///cout<<bravais_lattice<<endl;
	int reading_W=0;
	int number_integration_points=4;
	string file_macroscopic_dielectric_function_bse_name="bse_2x4_864_8bands.final_realspacedipoles.txt";
	htbse.pull_macroscopic_bse_dielectric_function(omegas_path,number_omegas_path,eta,file_macroscopic_dielectric_function_bse_name,lorentzian,tamn_dancoff,&coulomb_potential,&dielectric_function,adding_screening,order_approximation,number_integration_points,reading_W,ipa);
	
	return 1;
};