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

using namespace std;
using namespace arma;
const double pigreco = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679;
const double const_electron_charge = 1.602176634;
const double const_vacuum_dielectric_constant = 8.8541878128;
const double hbar = 6.582119569;
/// [hc]=eV*Ang 12.400 = Ry*Ang 911.38246268
const double hc = 911.38246268;
const double conversionNmtoeV = 6.2415064799632;

/// START DEFINITION DIFFERENT FUNCTIONS
vec function_vector_product(vec a, vec b)
{
	vec c(3);
	c(0) = a(1) * b(2) - a(2) * b(1);
	c(1) = a(2) * b(0) - a(0) * b(2);
	c(2) = a(0) * b(1) - a(1) * b(0);
	return c;
};
cx_mat function_building_exponential_factor(int htb_basis_dimension, field<mat> wannier_centers, int number_g_points_list, mat g_points_list, int number_k_points_list, int spinorial_calculation, vec excitonic_momentum){
	int htb_basis_dimension_2=htb_basis_dimension/2;
	cx_mat exponential_factor(htb_basis_dimension,number_g_points_list);
	if(spinorial_calculation==1){
		for(int spin_channel=0;spin_channel<2;spin_channel++)
			for(int g=0; g<number_g_points_list; g++){
				for(int i=0; i<htb_basis_dimension_2; i++){
					exponential_factor(spin_channel*htb_basis_dimension_2+i,g).real(cos(accu((wannier_centers(spin_channel)).col(i)%g_points_list.col(g))));
					exponential_factor(spin_channel*htb_basis_dimension_2+i,g).imag(sin(accu((wannier_centers(spin_channel)).col(i)%g_points_list.col(g))));
				}
			}
	}else{
		for(int g=0; g<number_g_points_list; g++)
			for(int i=0; i<htb_basis_dimension; i++){
				exponential_factor(i,g).real(cos(accu(wannier_centers(0).col(i)%g_points_list.col(g))));
				exponential_factor(i,g).imag(sin(accu(wannier_centers(0).col(i)%g_points_list.col(g))));
			}
	}
	return exponential_factor;
};
/// END DEFINITION DIFFERENT FUNCTIONS

/// START DEFINITION DIFFERENT CLASSES
/// Crystal_Lattice class
class Crystal_Lattice
{
private:
	int number_atoms;
	mat atoms_coordinates;
	mat bravais_lattice;
	double volume;

public:
	Crystal_Lattice()
	{
		number_atoms = 0;
		volume = 0.0;
	};
	void push_values(ifstream *bravais_lattice_file, ifstream *atoms_coordinates_file);
	void print();
	int pull_number_atoms()
	{
		return number_atoms;
	}
	vec pull_sitei_coordinates(int sitei)
	{
		vec atom_sitei_coordinates(3);
		atom_sitei_coordinates = atoms_coordinates(sitei);
		return atom_sitei_coordinates;
	}
	mat pull_bravais_lattice()
	{
		mat bravais_lattice_tmp(3, 3);
		bravais_lattice_tmp = bravais_lattice;
		return bravais_lattice_tmp;
	}
	mat pull_atoms_coordinates()
	{
		return atoms_coordinates;
	}
	double pull_volume()
	{
		return volume;
	}
};
void Crystal_Lattice::push_values(ifstream *bravais_lattice_file, ifstream *atoms_coordinates_file)
{
	if (bravais_lattice_file == NULL)
		throw std::invalid_argument("No Bravais Lattice file!");
	if (atoms_coordinates_file == NULL)
		throw std::invalid_argument("No Atoms Coordinateas file!");

	bravais_lattice.set_size(3, 3);
	bravais_lattice_file->seekg(0);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			*bravais_lattice_file >> bravais_lattice(i, j);
	atoms_coordinates_file->seekg(0);
	string line;
	while (atoms_coordinates_file->peek() != EOF)
	{
		getline(*atoms_coordinates_file, line);
		number_atoms++;
	}
	// cout<<"Number atoms "<<number_atoms<<endl;
	atoms_coordinates.set_size(3, number_atoms);
	atoms_coordinates_file->seekg(0);
	for (int i = 0; i < number_atoms; i++)
		for (int j = 0; j < 3; j++)
			*atoms_coordinates_file >> atoms_coordinates(j, i);

	volume = arma::det(bravais_lattice);
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
/// it is possible to define a list of k points directly from the BZ or give it as an input
/// in the class K points the points of FBZ are saved as k_points_list, while the points outside of the FBZ, defining the reciprocal lattice, are saved as g_points_list
class K_points
{
private:
	vec shift; 
	mat primitive_vectors; 
	double spacing_fbz_k_points_list;
	int number_k_points_list;
	mat k_points_list;
	int number_g_points_list;
	vec number_g_points_direction;
	mat g_points_list;
	double cutoff_g_points_list;
	int dimension_g_points_list;
	vec direction_cutting;
	mat bravais_lattice;
	double cell_volume;

public:
	K_points(Crystal_Lattice *crystal_lattice, vec shift_tmp)
	{
		spacing_fbz_k_points_list = 0;
		shift.zeros(3);

		number_k_points_list = 0;
		number_g_points_list = 0;
		cutoff_g_points_list = 0.0;

		primitive_vectors.set_size(3, 3);
		if (crystal_lattice == NULL)
			throw std::invalid_argument("Missing Crystal Lattice in K points grid building");
		
		shift = shift_tmp;
		bravais_lattice = crystal_lattice->pull_bravais_lattice();
		cell_volume=crystal_lattice->pull_volume();

		primitive_vectors.col(0) = function_vector_product(bravais_lattice.col(1), bravais_lattice.col(2));
		primitive_vectors.col(1) = -function_vector_product(bravais_lattice.col(2), bravais_lattice.col(0));
		primitive_vectors.col(2) = function_vector_product(bravais_lattice.col(0), bravais_lattice.col(1));
		double factor = 2 * pigreco / cell_volume;
		
		cout<<"Primitive Vectors:"<<endl;
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				primitive_vectors(i, j) = factor * primitive_vectors(i, j);
				cout<<primitive_vectors(i,j)<<" ";
			}
			cout<<endl;
		}
	}
	K_points(){
		spacing_fbz_k_points_list = 0;
		number_k_points_list = 0;
		number_g_points_list = 0;
		cutoff_g_points_list = 0.0;
	}
	void push_k_points_list_values_fbz(double spacing_fbz_k_points_list);
	void push_k_points_list_values(ifstream *k_points_list_file, int number_k_points_list)
	{
		number_k_points_list = number_k_points_list;
		k_points_list.set_size(3, number_k_points_list);
		k_points_list_file->seekg(0);
		int counting = 0;
		while (k_points_list_file->peek() != EOF)
		{
			if (counting < number_k_points_list)
			{
				*k_points_list_file >> k_points_list(0, counting);
				*k_points_list_file >> k_points_list(1, counting);
				*k_points_list_file >> k_points_list(2, counting);
				counting = counting + 1;
			}
			else
				break;
		}
		k_points_list_file->close();
	}
	void push_g_points_list_values(double cutoff_g_points_list_tmp, int dimension_g_points_list_tmp, vec direction_cutting_tmp);
	double pull_spacing_fbz_k_points_list(){
		return spacing_fbz_k_points_list;
	};
	vec pull_shift(){
		return shift;
	};
	mat pull_primitive_vectors()
	{
		return primitive_vectors;
	}
	mat pull_k_points_list_values()
	{
		return k_points_list;
	}
	mat pull_g_points_list_values()
	{
		return g_points_list;
	}
	int pull_number_k_points_list()
	{
		return number_k_points_list;
	}
	int pull_number_g_points_list()
	{
		return number_g_points_list;
	}
	void print();
	double pull_volume(){
		return cell_volume;
	}
};
void K_points::push_k_points_list_values_fbz(double spacing_fbz_k_points_list)
{
	spacing_fbz_k_points_list = spacing_fbz_k_points_list;

	vec bz_number_k_points(3);
	for (int i = 0; i < 3; i++)
		bz_number_k_points(i) = int(sqrt(accu(primitive_vectors.col(i) % primitive_vectors.col(i))) / spacing_fbz_k_points_list);

	int limiti = int(bz_number_k_points(0));
	int limitj = int(bz_number_k_points(1));
	int limitk = int(bz_number_k_points(2));
	number_k_points_list = limiti * limitj * limitk;
	k_points_list.set_size(3, number_k_points_list);
	
	cout << number_k_points_list << endl;
	int count = 0;
	for (int i = 0; i < limiti; i++)
		for (int j = 0; j < limitj; j++)
			for (int k = 0; k < limitk; k++)
			{
				for (int r = 0; r < 3; r++)
					k_points_list(r, count) = ((double)i / limiti) * (shift(r) + primitive_vectors(r, 0)) + ((double)j / limitj) * (shift(r) + primitive_vectors(r, 1)) + ((double)k / limitk) * (shift(r) + primitive_vectors(r, 2));
				count = count + 1;
			}
};
void K_points::print()
{
	cout << "K points list" << endl;
	for (int i = 0; i < number_k_points_list; i++)
	{
		cout << " ( ";
		for (int r = 0; r < 3; r++)
			cout << k_points_list(r, i) << " ";
		cout << " ) " << endl;
	}
	cout << "G points list" << endl;
	for (int i = 0; i < number_g_points_list; i++)
	{
		cout << " ( ";
		for (int r = 0; r < 3; r++)
			cout << g_points_list(r, i) << " ";
		cout << " ) " << endl;
	}
	
};
void K_points::push_g_points_list_values(double cutoff_g_points_list_tmp, int dimension_g_points_list_tmp, vec direction_cutting_tmp){
	dimension_g_points_list=dimension_g_points_list_tmp;
	direction_cutting=direction_cutting_tmp;
	cutoff_g_points_list=cutoff_g_points_list_tmp;

	double max_g_value=cutoff_g_points_list_tmp*1000/hc;
	cout<<"Calculating g values..."<<endl;
	//cout<<max_g_value<<endl;

	if(cutoff_g_points_list!=0){
		if(dimension_g_points_list==3){
			number_g_points_direction.zeros(3);
			number_g_points_list=1;
			for(int i=0;i<3;i++){
				number_g_points_direction(i)=int(max_g_value/norm(primitive_vectors.col(i),2));
				number_g_points_list=number_g_points_list*(2*number_g_points_direction(i)+1);
				cout<<number_g_points_direction(i)<<endl;
			}
			g_points_list.set_size(3,number_g_points_list);
			cout << number_g_points_list << endl;
			int count = 0;
			for (int i = -number_g_points_direction(0); i <= number_g_points_direction(0); i++)
				for (int j = -number_g_points_direction(1); j <= number_g_points_direction(1); j++)
					for (int k = -number_g_points_direction(2); k <= number_g_points_direction(2); k++)
					{
						for (int r = 0; r < 3; r++)
							g_points_list(r, count) = i * (shift(r) + primitive_vectors(r, 0)) + j * (shift(r) + primitive_vectors(r, 1)) + k * (shift(r) + primitive_vectors(r, 2));
						count = count + 1;
					}
		}else if(dimension_g_points_list==2){
			number_g_points_direction.zeros(2);
			mat reciprocal_plane_along; reciprocal_plane_along.zeros(3,2);
			int count=0;
			number_g_points_list=1;
			for(int i=0;i<3;i++)
				if(direction_cutting(i)==1){
					reciprocal_plane_along.col(count)=primitive_vectors.col(i);
					number_g_points_direction(count)=int(max_g_value/norm(primitive_vectors.col(i),2));
					number_g_points_list=number_g_points_list*(2*number_g_points_direction(i)+1);
					cout<<number_g_points_direction(count)<<endl;
					count++;
				}
			g_points_list.set_size(3,number_g_points_list);
			//cout << number_g_points_list << endl;
			count = 0;
			for (int i = -number_g_points_direction(0); i <= number_g_points_direction(0); i++)
				for (int j = -number_g_points_direction(1); j <= number_g_points_direction(1); j++){
					for (int r = 0; r < 3; r++)
						g_points_list(r, count) = i * (shift(r) + reciprocal_plane_along(r, 0)) + j * (shift(r) + reciprocal_plane_along(r, 1));
					//cout<<g_points_list.col(count)<<endl;
					count = count + 1;
				}
		}else
			cout<<"Not implemented case"<<endl;
	}else{
		number_g_points_list=1;
		g_points_list.zeros(3,number_g_points_list);
	}
	cout<<"Number g points: "<<number_g_points_list<<endl;
};	

/// Hamiltonian_TB class
class Hamiltonian_TB
{
private:
	int spinorial_calculation;
	int number_wannier_functions;
	int htb_basis_dimension;
	int number_primitive_cells;
	vec weights_primitive_cells;
	mat positions_primitive_cells;
	double fermi_energy;
	field<cx_cube> hamiltonian;
	field<mat> wannier_centers; 
	cx_mat exponential_factor; 

public:
	Hamiltonian_TB()
	{
		number_wannier_functions = 0;
		htb_basis_dimension = 0;
		spinorial_calculation = 0;
		fermi_energy = 0;
		number_primitive_cells = 0;
	}
	/// reading hamiltonian from wannier90 output
	void push_values(ifstream *wannier90_hr_file, ifstream *wannier90_centers_file, double fermi_energy, int spinorial_calculation, int number_atoms);
	field<cx_cube> pull_hamiltonian()
	{
		return hamiltonian;
	}
	int pull_htb_basis_dimension()
	{
		return htb_basis_dimension;
	}
	int pull_number_wannier_functions()
	{
		return number_wannier_functions;
	}
	double pull_fermi_energy(){
		return fermi_energy;
	}
	field<cx_mat> FFT(vec k_point);
	tuple<mat, cx_mat> pull_ks_states(vec k_point);
	tuple<mat, cx_mat> pull_ks_states_subset(vec k_point, int minimum_valence, int maximum_conduction);
	void print_hamiltonian()
	{
		cout << "Printing Hamiltonian..." << endl;
		int spin_counting = 0;
		while (spin_counting < 2)
		{
			for (int i = 0; i < number_primitive_cells; i++)
				for (int q = 0; q < number_wannier_functions; q++)
					for (int s = 0; s < number_wannier_functions; s++)
					{
						for (int r = 0; r < 3; r++)
							cout << positions_primitive_cells(r, i) << " ";
						cout << q << " " << s << " " << hamiltonian(spin_counting)(q, s, i) << endl;
					}
			for(int i=0;i<number_wannier_functions;i++)
			{
				for (int r = 0; r < 3; r++)
					cout << wannier_centers(spin_counting)(r,i) <<" ";
				cout<<endl;
			}
			if (spinorial_calculation == 1)
				spin_counting = spin_counting + 1;
			else
				break;
		}
	}
	void print_ks_states(vec k_point, int minimum_valence, int maximum_conduction);
	cx_mat extract_usual_dipoles(vec k_point, int minimum_valence, int maximum_conduction, double eta);
	field<mat> pull_wannier_centers(){
		return wannier_centers;
	};
};
void Hamiltonian_TB::push_values(ifstream *wannier90_hr_file,  ifstream *wannier90_centers_file, double fermi_energy_tmp, int spinorial_calculation_tmp, int number_atoms)
{
	cout << "Be Carefull: if you are doing a collinear spin calculation, the number of Wannier functions in the two spin channels has to be the same!!" << endl;
	fermi_energy = fermi_energy_tmp;
	spinorial_calculation = spinorial_calculation_tmp;
	if (wannier90_hr_file == NULL)
	{
		throw std::invalid_argument("No Wannier90 Hamiltonian file!");
	}
	else
	{
		cout << "Reading Hamiltonian..." << endl;
		wannier90_hr_file->seekg(0);
	}
	

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
	/// the Hamiltonians for the spinorial calculation = 1, should be one under the other(all the hr FILE (time included))
	while (wannier90_hr_file->peek() != EOF && spin_channel < 2)
	{
		getline(*wannier90_hr_file >> ws, history_time);
		*wannier90_hr_file >> number_wannier_functions;
		*wannier90_hr_file >> number_primitive_cells;
		// cout<<"Number wannier functions "<<number_wannier_functions<<endl;
		// cout<<"Number primitive cells "<<number_primitive_cells<<endl;
		if (spin_channel == 0)
		{
			// initialization of the vriables
			weights_primitive_cells.set_size(number_primitive_cells);
			positions_primitive_cells.set_size(3, number_primitive_cells);
			if (spinorial_calculation == 1)
			{
				/// two loops
				hamiltonian.set_size(2);
				hamiltonian(0).set_size(number_wannier_functions, number_wannier_functions, number_primitive_cells);
				hamiltonian(1).set_size(number_wannier_functions, number_wannier_functions, number_primitive_cells);
				htb_basis_dimension = number_wannier_functions * 2;
			}
			else
			{
				/// one loop
				hamiltonian.set_size(1);
				hamiltonian(0).set_size(number_wannier_functions, number_wannier_functions, number_primitive_cells);
				htb_basis_dimension = number_wannier_functions;
			}
		}
		total_elements = number_wannier_functions * number_wannier_functions * number_primitive_cells;
		// cout<<"Total elements "<<total_elements<<endl;
		counting_positions = 0;
		while (counting_positions < number_primitive_cells)
		{
			*wannier90_hr_file >> weights_primitive_cells(counting_positions);
			// cout<<counting_positions<<" "<<weights_primitive_cells(counting_positions)<<" ";
			counting_positions++;
		}
		counting_primitive_cells = 0;
		counting_positions = 0;
		/// the hamiltonian in the collinear case is diagonal in the spin channel
		while (counting_positions < total_elements)
		{
			if (counting_positions == number_wannier_functions * number_wannier_functions * counting_primitive_cells)
			{
				*wannier90_hr_file >> positions_primitive_cells(0, counting_primitive_cells) >> positions_primitive_cells(1, counting_primitive_cells) >> positions_primitive_cells(2, counting_primitive_cells) >> l >> m >> real_part >> imag_part;
				counting_primitive_cells = counting_primitive_cells + 1;
			}
			else
			{
				*wannier90_hr_file >> trashing_positions[0] >> trashing_positions[1] >> trashing_positions[2] >> l >> m >> real_part >> imag_part;
			}
			real_part = real_part * weights_primitive_cells(counting_primitive_cells - 1);
			imag_part = imag_part * weights_primitive_cells(counting_primitive_cells - 1);
			hamiltonian(spin_channel)(l - 1, m - 1, counting_primitive_cells - 1).real(real_part);
			hamiltonian(spin_channel)(l - 1, m - 1, counting_primitive_cells - 1).imag(imag_part);
			// cout<<spin_channel<<l<<" "<<m<<" "<<real_part<<" "<<imag_part<<" "<<counting_positions<<" "<<total_elements<<" "<<counting_primitive_cells<<" "<<number_primitive_cells<<endl;
			counting_positions++;
		}
		// cout<<spinorial_calculation<<" "<<number_wannier_functions<<" "<<number_primitive_cells<<" "<<spin_channel<<" "<<endl;
		spin_channel = spin_channel + 1;
	}
	cout << "Hamiltonian saved." << endl;
	
	if (wannier90_centers_file == NULL)
	{
		throw std::invalid_argument("No Wannier90 Centers file!");
	}
	else
	{
		cout << "Reading Centers..." << endl;
		wannier90_centers_file->seekg(0);
	}
	char element_name;
	int number_lines;
	spin_channel=0;
	while (wannier90_centers_file->peek() != EOF && spin_channel < 2)
	{
		if (spin_channel == 0)
		{
			// initialization of the variables
			if (spinorial_calculation == 1)
			{
				/// two loops
				wannier_centers.set_size(2);
				wannier_centers(0).set_size(3,number_wannier_functions);
				wannier_centers(1).set_size(3,number_wannier_functions);
			}
			else
			{
				/// one loop
				wannier_centers.set_size(1);
				wannier_centers(0).set_size(3,number_wannier_functions);
			}
		}

		*wannier90_centers_file >> number_lines;
		getline(*wannier90_centers_file >> ws, history_time);
		counting_positions = 0;
		while (counting_positions < number_wannier_functions)
		{
			*wannier90_centers_file >> element_name>>wannier_centers(spin_channel)(0,counting_positions) >> wannier_centers(spin_channel)(1,counting_positions) >> wannier_centers(spin_channel)(2,counting_positions);
			counting_positions++;
		}
		counting_positions = 0;
		while (counting_positions < number_atoms)
		{
			getline(*wannier90_centers_file >> ws, trashing_lines);
			counting_positions++;
		}
		spin_channel++;
	}
	cout << "Centers saved." << endl;
};
field<cx_mat> Hamiltonian_TB::FFT(vec k_point)
{
	//cout << "Fourier transforming..." << endl;

	int flag_spin_channel = 0;
	int offset;
	field<cx_mat> fft_hamiltonian;
	vec temporary_cos(number_primitive_cells);
	vec temporary_sin(number_primitive_cells);
	vec real_part_hamiltonian(number_primitive_cells);
	vec imag_part_hamiltonian(number_primitive_cells);
	for (int r = 0; r < number_primitive_cells; r++)
	{
		temporary_cos(r) = cos(accu(k_point % positions_primitive_cells.col(r)));
		temporary_sin(r) = sin(accu(k_point % positions_primitive_cells.col(r)));
		// cout<<temporary_cos(r)<<" "<<temporary_sin(r)<<endl;
	}
	if (spinorial_calculation == 1)
	{
		fft_hamiltonian.set_size(2);
		fft_hamiltonian(0).set_size(number_wannier_functions, number_wannier_functions);
		fft_hamiltonian(1).set_size(number_wannier_functions, number_wannier_functions);
		while (flag_spin_channel < 2)
		{
			offset = number_wannier_functions * flag_spin_channel;
			for (int l = 0; l < number_wannier_functions; l++)
			{
				for (int m = 0; m < number_wannier_functions; m++)
				{
					real_part_hamiltonian = real(hamiltonian(flag_spin_channel).tube(l, m));
					imag_part_hamiltonian = imag(hamiltonian(flag_spin_channel).tube(l, m));
					fft_hamiltonian(flag_spin_channel)(l, m).real(accu(real_part_hamiltonian % temporary_cos) - accu(imag_part_hamiltonian % temporary_sin));
					fft_hamiltonian(flag_spin_channel)(l, m).imag(accu(imag_part_hamiltonian % temporary_cos) + accu(real_part_hamiltonian % temporary_sin));
					// cout<<fft_hamiltonian(flag_spin_channel)(l,m)<<" ";
				}
				// cout<<endl;
			}
			flag_spin_channel++;
		}
	}
	else
	{
		fft_hamiltonian.set_size(1);
		fft_hamiltonian(0).set_size(number_wannier_functions, number_wannier_functions);
		for (int l = 0; l < number_wannier_functions; l++)
		{
			for (int m = 0; m < number_wannier_functions; m++)
			{
				real_part_hamiltonian = real(hamiltonian(0).tube(l, m));
				imag_part_hamiltonian = imag(hamiltonian(0).tube(l, m));
				fft_hamiltonian(0)(l, m).real(accu(real_part_hamiltonian % temporary_cos) - accu(imag_part_hamiltonian % temporary_sin));
				fft_hamiltonian(0)(l, m).imag(accu(imag_part_hamiltonian % temporary_cos) + accu(real_part_hamiltonian % temporary_sin));
				// cout<<fft_hamiltonian(flag_spin_channel)(l,m)<<" ";
			}
			// cout<<endl;
		}
	}
	//cout << "Finished Fourier transforming." << endl;
	return fft_hamiltonian;
};
tuple<mat, cx_mat> Hamiltonian_TB::pull_ks_states(vec k_point)
{
	cx_double temporary_norm;
	/// the eigenvalues are saved into a two component element, in order to make the code more general
	mat ks_eigenvalues_spinor(2, number_wannier_functions, fill::zeros);
	cx_mat ks_eigenvectors(htb_basis_dimension, htb_basis_dimension);

	field<cx_mat> fft_hamiltonian = FFT(k_point);
	//cout << "Diagonalizing" << endl;

	if (spinorial_calculation == 1)
	{
		vec eigenvalues_up;
		cx_mat eigenvectors_up;
		vec eigenvalues_down;
		cx_mat eigenvectors_down;
		cx_mat fft_hamiltonian_up = fft_hamiltonian(0);
		cx_mat fft_hamiltonian_down = fft_hamiltonian(1);
		eig_sym(eigenvalues_up, eigenvectors_up, fft_hamiltonian_up);
		eig_sym(eigenvalues_down, eigenvectors_down, fft_hamiltonian_down);

		vec ks_eigenvalues_up = sort(eigenvalues_up);
		vec ks_eigenvalues_down = sort(eigenvalues_down);
		mat ordering_up(number_wannier_functions, 2, fill::zeros);
		mat ordering_down(number_wannier_functions, 2, fill::zeros);
		for (int i = 0; i < number_wannier_functions; i++)
		{
			if (ordering_up(i, 1) != 1)
				for (int j = 0; j < number_wannier_functions; j++)
					if (ks_eigenvalues_up(i) == eigenvalues_up(j))
					{
						ordering_up(i, 0) = j;
						ordering_up(i, 1) = 1;
					}
			if (ordering_down(i, 1) != 1)
				for (int j = 0; j < number_wannier_functions; j++)
					if (ks_eigenvalues_down(i) == eigenvalues_down(j))
					{
						ordering_down(i, 0) = j;
						ordering_down(i, 1) = 1;
					}
		}
		//cout << "Finished ordering " << endl;
		/// in the case of spinorial_calculation=1 combining the two components of spin into a single spinor
		/// saving the ordered eigenvectors in the matrix ks_eigenvectors_spinor
		cx_mat ks_eigenvectors_spinor(htb_basis_dimension, number_wannier_functions);
		for (int i = 0; i < number_wannier_functions; i++)
		{
			for (int j = 0; j < htb_basis_dimension; j++)
			{
				if (j < number_wannier_functions)
					ks_eigenvectors_spinor(j, i) = eigenvectors_up(j, ordering_up(i, 0));
				else
					ks_eigenvectors_spinor(j, i) = eigenvectors_down(j - number_wannier_functions, ordering_down(i, 0));
			}
			temporary_norm = norm(ks_eigenvectors_spinor.col(i), 2);
			ks_eigenvectors_spinor.col(i) = ks_eigenvectors_spinor.col(i) / temporary_norm;
			for (int r = 0; r < 2; r++)
				ks_eigenvalues_spinor(r, i) = (1 - r) * ks_eigenvalues_up(i) + r * ks_eigenvalues_down(i);
		}
		///cout << "Finished extraction " << endl;
		return {ks_eigenvalues_spinor, ks_eigenvectors_spinor};
	}
	else
	{
		vec eigenvalues;
		cx_mat eigenvectors;
		eig_sym(eigenvalues, eigenvectors, fft_hamiltonian(0));
		vec ks_eigenvalues = sort(eigenvalues);
		/// function to order eigenvectors
		mat ordering(htb_basis_dimension, 2, fill::zeros);
		for (int i = 0; i < htb_basis_dimension; i++)
			if (ordering(i, 1) != 1)
				for (int j = 0; j < htb_basis_dimension; j++)
					if (ks_eigenvalues(i) == eigenvalues(j))
					{
						ordering(i, 0) = j;
						ordering(i, 1) = 1;
					}
		/// ordering respectively the eigenvectors and normalizing them
		for (int i = 0; i < htb_basis_dimension; i++)
		{
			temporary_norm = norm(eigenvectors.col(ordering(i, 0)), 2);
			ks_eigenvectors.col(i) = eigenvectors.col(ordering(i, 0)) / temporary_norm;
		}
		for (int i = 0; i < number_wannier_functions; i++)
		{
			ks_eigenvalues_spinor(0, i) = ks_eigenvalues(i);
			ks_eigenvalues_spinor(1, i) = ks_eigenvalues(i);
		}
		return {ks_eigenvalues_spinor, ks_eigenvectors};
	}
};
tuple<mat, cx_mat> Hamiltonian_TB::pull_ks_states_subset(vec k_point, int minimum_valence, int maximum_conduction)
{
	int number_valence_bands = 0;
	int number_conduction_bands = 0;
	int dimensions_subspace = minimum_valence + maximum_conduction;
	tuple<mat, cx_mat> ks_states;
	ks_states = pull_ks_states(k_point);
	mat ks_eigenvalues = get<0>(ks_states);
	cx_mat ks_eigenvectors = get<1>(ks_states);

	/// distinguishing between valence and conduction states
	//cout << "Extracting subset " << number_valence_bands << " " << number_conduction_bands << endl;
	for (int i = 0; i < number_wannier_functions; i++)
	{
		//cout << ks_eigenvalues(0, i) << " " << ks_eigenvalues(1, i) << endl;
		if (ks_eigenvalues(0, i) <= fermi_energy && ks_eigenvalues(1, i) <= fermi_energy)
			number_valence_bands++;
		else
			number_conduction_bands++;
	}

	//cout << " " << number_valence_bands << " " << number_conduction_bands << endl;
	mat ks_eigenvalues_subset(2, dimensions_subspace);

	if (number_conduction_bands < maximum_conduction)
		throw std::invalid_argument("Too many conduction bands required");
	if (number_valence_bands < minimum_valence)
		throw std::invalid_argument("Too many valence bands required");

	/// first are written valence states, than (at higher rows) conduction states
	cx_mat ks_eigenvectors_subset(htb_basis_dimension, dimensions_subspace);

	for (int i = 0; i < dimensions_subspace; i++)
	{
		if (i < minimum_valence)
		{
			ks_eigenvectors_subset.col(i) = ks_eigenvectors.col((number_valence_bands - 1) - i);
			ks_eigenvalues_subset.col(i) = ks_eigenvalues.col((number_valence_bands - 1) - i);
		}
		else
		{
			ks_eigenvectors_subset.col(i) = ks_eigenvectors.col(number_valence_bands + (i - minimum_valence));
			ks_eigenvalues_subset.col(i) = ks_eigenvalues.col(number_valence_bands + (i - minimum_valence));
		}
	}
	return {ks_eigenvalues_subset, ks_eigenvectors_subset};
};
void Hamiltonian_TB::print_ks_states(vec k_point, int minimum_valence, int maximum_conduction)
{
	tuple<mat, cx_mat> results_htb;
	tuple<mat, cx_mat> results_htb_subset;
	results_htb = pull_ks_states(k_point);
	results_htb_subset = pull_ks_states_subset(k_point, minimum_valence, maximum_conduction);
	mat eigenvalues = get<0>(results_htb);
	cx_mat eigenvectors = get<1>(results_htb);
	mat eigenvalues_subset = get<0>(results_htb_subset);
	cx_mat eigenvectors_subset = get<1>(results_htb_subset);

	////In the case of spinorial_calculation=1, the spinor components of each wannier function are one afte the other
	////i.e. wannier function 0 : spin up component columnt 0, spin down component column 1...and so on...
	////moreover, in the case of ks_subset the valence states are written before the conduction states
	cout << "All bands" << endl;
	for (int i = 0; i < number_wannier_functions; i++)
	{
		printf("%d	%.5f %.5f\n", i, eigenvalues(0, i), eigenvalues(1, i));
		for (int j = 0; j < htb_basis_dimension; j++)
			printf("(%.5f,%.5f)", real(eigenvectors(j, i)), imag(eigenvectors(j, i)));
		cout << endl;
	}
	cout << "Only subset" << endl;
	for (int i = 0; i < minimum_valence + maximum_conduction; i++)
	{
		printf("%d	%.5f %.5f\n", i, eigenvalues_subset(0, i), eigenvalues_subset(1, i));
		for (int j = 0; j < htb_basis_dimension; j++)
			printf("(%.5f,%.5f)", real(eigenvectors_subset(j, i)), imag(eigenvectors_subset(j, i)));
		cout << endl;
	}
};
cx_mat Hamiltonian_TB::extract_usual_dipoles(vec k_point, int number_valence_bands, int number_conduction_bands, double eta){
	/// i am going to transform the dipoles in terms of the wannier functions into dipoles in terms of KS states
	/// p_{mn}=\sum_{ij}<mi>p_{ij}<jn>
	/// it is sufficient to do the vector x matrix x vector product
	int number_valence_plus_conduction=number_valence_bands+number_conduction_bands;
	int number_valence_times_conduction=number_valence_bands*number_conduction_bands;
	tuple<mat,cx_mat> ks_states;
	ks_states = pull_ks_states_subset(k_point,number_valence_bands,number_conduction_bands);
	mat ks_energies_k_point = get<0>(ks_states);
	cx_mat ks_states_k_point = get<1>(ks_states);
	cx_vec temporary_vector_0(number_wannier_functions);
	vec exponential_factor_cos(number_primitive_cells);
	vec exponential_factor_sin(number_primitive_cells);

	for (int r = 0; r < number_primitive_cells; r++){
		exponential_factor_cos(r)=cos(accu(k_point%positions_primitive_cells.col(r)));
		exponential_factor_sin(r)=sin(accu(k_point%positions_primitive_cells.col(r)));
	}
	vec temporary_vector_re(number_primitive_cells);
	vec temporary_vector_im(number_primitive_cells);
	cx_vec temporary_vector_hamiltonian(number_primitive_cells);
	cx_double ratio;

	cx_mat dipoles;
	cx_mat dipoles_reduced;
	cout<<"start dipoles calculation"<<endl;
	if(spinorial_calculation==1){
		dipoles.zeros(number_wannier_functions,number_wannier_functions);
		dipoles_reduced.zeros(2*number_valence_times_conduction,3);
		int q=0;
		for(int xyz=0;xyz<3;xyz++)
			for(int spin_channel=0;spin_channel<2;spin_channel++){
				for(int w1=0;w1<number_wannier_functions;w1++)
					for(int w2=0;w2<number_wannier_functions;w2++){
						temporary_vector_hamiltonian=hamiltonian(spin_channel).tube(w1,w2);
						temporary_vector_re=((positions_primitive_cells.row(xyz)).t())%real(temporary_vector_hamiltonian);
						temporary_vector_im=((positions_primitive_cells.row(xyz)).t())%imag(temporary_vector_hamiltonian);
						dipoles(w1,w2).real(
							+accu(exponential_factor_sin%temporary_vector_im)
							+accu(exponential_factor_cos%temporary_vector_re));
						dipoles(w1,w2).imag(
							+accu(exponential_factor_cos%temporary_vector_im)
							-accu(exponential_factor_sin%temporary_vector_re));
					}
				q=0;
				for(int n=0;n<number_conduction_bands;n++)
					for(int m=0;m<number_valence_bands;m++){
						ratio=1/(pow(ks_energies_k_point(spin_channel,n+number_valence_bands)-ks_energies_k_point(spin_channel,m),2)+pow(eta,2));
						ratio.real(ks_energies_k_point(spin_channel,n+number_valence_bands)-ks_energies_k_point(spin_channel,m));
						ratio.imag(-eta);
						for(int j=0;j<number_wannier_functions;j++)
							temporary_vector_0(j)=accu(dipoles.col(j)%conj(ks_states_k_point.submat(spin_channel*number_wannier_functions,n+number_valence_bands,(1+spin_channel)*number_wannier_functions-1,n+number_valence_bands)));
						dipoles_reduced(spin_channel*number_valence_times_conduction+q,xyz)=ratio*accu(temporary_vector_0%ks_states_k_point.submat(spin_channel*number_wannier_functions,m,(1+spin_channel)*number_wannier_functions-1,m));
						q++;
					}
			}
		cout<<"end dipoles calculation"<<endl;
		return dipoles_reduced;
	}else{
		dipoles.zeros(number_wannier_functions,number_wannier_functions);
		dipoles_reduced.zeros(number_valence_times_conduction,3);
		int q=0;
		for(int xyz=0;xyz<3;xyz++){
			for(int w1=0;w1<number_wannier_functions;w1++)
				for(int w2=0;w2<number_wannier_functions;w2++){
						temporary_vector_hamiltonian=hamiltonian(0).tube(w1,w2);
						temporary_vector_re=((positions_primitive_cells.row(xyz)).t())%real(temporary_vector_hamiltonian);
						temporary_vector_im=((positions_primitive_cells.row(xyz)).t())%imag(temporary_vector_hamiltonian);
						dipoles(w1,w2).real(
							+accu(exponential_factor_sin%temporary_vector_im)
							+accu(exponential_factor_cos%temporary_vector_re));
						dipoles(w1,w2).imag(
							+accu(exponential_factor_cos%temporary_vector_im)
							-accu(exponential_factor_sin%temporary_vector_re));
					}
			q=0;
			for(int n=0;n<number_conduction_bands;n++)
				for(int m=0;m<number_valence_bands;m++){
					ratio=1/(pow(ks_energies_k_point(0,n+number_valence_bands)-ks_energies_k_point(0),m),2)+pow(eta,2);
					ratio.real(ks_energies_k_point(0,n+number_valence_bands)-ks_energies_k_point(0,m));
					ratio.imag(-eta);
					for(int j=0;j<number_wannier_functions;j++)
						temporary_vector_0(j)=accu(dipoles.col(j)%conj(ks_states_k_point.col(n+number_valence_bands)));
					dipoles_reduced(q,xyz)=accu(temporary_vector_0%ks_states_k_point.col(m));
					q++;
				}
		}
		cout<<"end dipoles calculation"<<endl;
		return dipoles_reduced;
	}
};

// Coulomb_Potential class
class Coulomb_Potential
{
private:
	double cell_volume;
	double electron_charge;
	double effective_dielectric_constant;
	double vacuum_dielectric_constant;
	double minimum_k_point_modulus;
	mat primitive_vectors;
	vec direction_cutting;
	int dimension_potential;
	K_points *k_points;

public:
	Coulomb_Potential(){
		cell_volume=0.0;
		electron_charge=0.0;
		effective_dielectric_constant=0.0;
		vacuum_dielectric_constant=0.0;
		minimum_k_point_modulus=0.0;
		dimension_potential=0.0;
	};
	void push_values(K_points* k_points_tmp, double effective_dielectric_constant_tmp, double minimum_k_point_modulus_tmp, int dimension_potential_tmp, vec direction_cutting_tmp)
	///direction of the bravais lattice along whic the cut is considered (1 is cut and 0 is no-cut)
	{
		k_points=k_points_tmp;
		cell_volume = k_points->pull_volume();
		effective_dielectric_constant = effective_dielectric_constant_tmp;
		minimum_k_point_modulus = minimum_k_point_modulus_tmp;
		electron_charge = const_electron_charge;
		vacuum_dielectric_constant = const_vacuum_dielectric_constant;
		dimension_potential = dimension_potential_tmp;
		primitive_vectors = k_points->pull_primitive_vectors();
		direction_cutting = direction_cutting_tmp;
	};
	double pull(vec k_point);
	double pull_cell_volume()
	{
		return cell_volume;
	};
	void print(){
		cout<<"Volume: "<<cell_volume<<endl;
		cout<<"Electron charge: "<<electron_charge<<endl;
		cout<<"Effective dielectric constant: "<<effective_dielectric_constant<<endl;
		cout<<"Vacuum dielectric constant: "<<vacuum_dielectric_constant<<endl;
		cout<<"Minimum k point modulus: "<<minimum_k_point_modulus<<endl;
		cout<<"Primitive vectors: "<<endl;
		cout<<primitive_vectors<<endl;
		cout<<"Direction cutting"<<endl;
		cout<<direction_cutting<<endl;
		cout<<"Dimension potential: "<<dimension_potential<<endl;
		cout<<"K points address: "<<k_points<<endl;
	};
	void print_profile(mat k_points_path, int number_k_points){
		for(int i=0;i<number_k_points;i++)
			cout<<norm(k_points_path.col(i))<<" "<<pull(k_points_path.col(i))<<endl;
	};
};
double Coulomb_Potential::pull(vec k_point)
{
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
			coulomb_potential = -conversionNmtoeV * 100 * pow(electron_charge, 2) / (2 * cell_volume * vacuum_dielectric_constant * effective_dielectric_constant * pow(modulus_k_point, 2));
	}else if(dimension_potential==2){
		vec primitive_along(3);
		for(int i=0;i<3;i++)
			if(direction_cutting(i)==0){
				primitive_along=primitive_vectors.col(i);
			}
		vec k_point_orthogonal; k_point_orthogonal.zeros(3);
		vec k_point_along; k_point_along.zeros(3);
		vec unity; unity.ones(3);
		k_point_along=(primitive_along%k_point)/norm(primitive_along);
		k_point_orthogonal=k_point-k_point_along;
		if (modulus_k_point < minimum_k_point_modulus)
			coulomb_potential=0;
		else{
			coulomb_potential = -conversionNmtoeV * 100 * pow(electron_charge, 2) / (2 * cell_volume * vacuum_dielectric_constant * effective_dielectric_constant * pow(modulus_k_point, 2));
			coulomb_potential = coulomb_potential * (1-exp(-norm(k_point_orthogonal,2)*norm(primitive_along)/2)*cos(norm(k_point_along)*norm(primitive_along)));
		}
	}else
		cout<<"Not implemented case ciao"<<endl;

	return coulomb_potential;
};
/// (non-interacting) Green Function
class KS_Green_Function
{
private:
	int spinorial_calculation;
	int number_wannier_functions;
	int number_atoms;
	double fermi_energy;
	double eta;
	field<cx_cube> hamiltonian;
	K_points k_points;
	Hamiltonian_TB htb;

public:
	KS_Green_Function(){
		spinorial_calculation=0;
		number_wannier_functions=0;
		fermi_energy=0;
	}
	void push_values(ifstream *wannier90_hr_file,  ifstream *wannier90_centers_file, double fermi_energy_tmp, int spinorial_calculation_tmp, K_points k_points_tmp, double eta_tmp);
	field<cx_cube> pull_ks_green_k_space(double energy, vec k_points_shift);
	int pull_number_wannier_functions(){
		return number_wannier_functions;
	}
};
void KS_Green_Function:: push_values(ifstream *wannier90_hr_file, ifstream *wannier90_centers_file, double fermi_energy_tmp, int spinorial_calculation_tmp, K_points k_points_tmp,double eta_tmp){
	spinorial_calculation=spinorial_calculation_tmp;
	fermi_energy=fermi_energy_tmp;
	eta=eta_tmp;
	htb.push_values(wannier90_hr_file, wannier90_centers_file, fermi_energy_tmp, spinorial_calculation, number_atoms);	
	number_wannier_functions=htb.pull_number_wannier_functions();
	k_points=k_points_tmp;
};
field<cx_cube> KS_Green_Function:: pull_ks_green_k_space(double energy, vec k_points_shift){
	int number_k_points_list=k_points.pull_number_k_points_list();
	vec k_point(3);
	tuple<mat,cx_mat> ks_states;
	mat ks_energies_k_point; ks_energies_k_point.set_size(2,number_wannier_functions);
	cx_double ieta;	ieta.imag(eta);

	if(spinorial_calculation==1){
		field<cx_cube> ks_green(2);
		ks_green(0).set_size(number_k_points_list,number_wannier_functions,number_wannier_functions);
		ks_green(1).set_size(number_k_points_list,number_wannier_functions,number_wannier_functions);
		for (int i = 0; i < number_k_points_list; i++)
		{
			k_point = (k_points.pull_k_points_list_values()).col(i)+k_points_shift;
			ks_states = htb.pull_ks_states(k_point);
			ks_energies_k_point = get<0>(ks_states);
			for(int spin_channel=0;spin_channel<2;spin_channel++)
				ks_green(spin_channel).slice(i)=
					diagmat(1/(energy-ks_energies_k_point.col(spin_channel)+ieta));
		}
		return ks_green; 
	}else{
		field<cx_cube> ks_green(1);
		ks_green(0).set_size(number_k_points_list,number_wannier_functions,number_wannier_functions);
		for (int i = 0; i < number_k_points_list; i++)
		{
			k_point = (k_points.pull_k_points_list_values()).col(i)+k_points_shift;
			ks_states = htb.pull_ks_states(k_point);
			ks_energies_k_point = get<0>(ks_states);
			ks_green(0).slice(i)=
					diagmat(1/(energy-ks_energies_k_point.col(0)+ieta));
		}
		return ks_green;
	}
};
/// Screening matrix elements
class Dipole_Elements
{
private:
	int number_k_points_list;
	int number_g_points_list;
	int htb_basis_dimension;
	int number_wannier_centers;
	field<mat> wannier_centers;
	vec excitonic_momentum;
	mat k_points_list;
	mat g_points_list;
	int number_valence_bands;
	int spin_number_valence_bands;
	int number_conduction_bands;
	int spin_number_conduction_bands;
	int number_valence_plus_conduction;
	int spin_number_valence_plus_conduction;
	Hamiltonian_TB *hamiltonian_tb;
	int spinorial_calculation;
	cx_mat exponential_factor;
public:
	Dipole_Elements(){
		number_k_points_list=0;
		number_g_points_list=0;
		htb_basis_dimension=0;
		number_conduction_bands=0;
		number_valence_bands=0;
	};
	void push_values(int number_k_points_list_tmp, mat k_points_list_tmp, int number_g_points_list_tmp, mat g_points_list_tmp, int number_wannier_centers_tmp, int number_valence_bands_tmp, int number_conduction_bands_tmp, Hamiltonian_TB *hamiltonian_tb_tmp, int spinorial_calculation_tmp){
		number_g_points_list=number_g_points_list_tmp;
		number_conduction_bands=number_conduction_bands_tmp;
		number_valence_bands=number_valence_bands_tmp;
		number_k_points_list=number_k_points_list_tmp;
		number_wannier_centers=number_wannier_centers_tmp;
		number_valence_plus_conduction=number_valence_bands_tmp+number_conduction_bands_tmp;
		k_points_list=k_points_list_tmp;
		g_points_list=g_points_list_tmp;
		hamiltonian_tb=hamiltonian_tb_tmp;
		spinorial_calculation=spinorial_calculation_tmp;
		htb_basis_dimension=hamiltonian_tb->pull_htb_basis_dimension();
		wannier_centers=hamiltonian_tb->pull_wannier_centers();
		exponential_factor=function_building_exponential_factor(htb_basis_dimension, wannier_centers, number_g_points_list, g_points_list, number_k_points_list, spinorial_calculation, excitonic_momentum);
	};
	tuple<mat,cx_mat> pull_values(vec excitonic_momentum);
	cx_mat pull_reduced_values_vc(vec excitonic_momentum, cx_mat rho);
	cx_mat pull_reduced_values_cc_vv(vec excitonic_momentum, cx_mat rho, int conduction_or_valence);
	////term 0: total, term 1: vc, term 2: cc term 3: vv
	void print(vec excitonic_momentum, int which_term){
		//for(int g=0;g<number_g_points_list;g++)
		//	cout<<exponential_factor.col(g)<<endl;
		
		tuple<mat,cx_mat> energies_and_dipole_elements=pull_values(excitonic_momentum);
		mat energies; energies=get<0>(energies_and_dipole_elements);
		cx_mat dipole_elements;	dipole_elements=get<1>(energies_and_dipole_elements);

		int states1;
		if(which_term==0){
			states1=spin_number_valence_plus_conduction*number_valence_plus_conduction*number_k_points_list;
		}else if(which_term==1){
			dipole_elements=pull_reduced_values_vc(excitonic_momentum,dipole_elements);
			states1=spin_number_valence_bands*number_conduction_bands*number_k_points_list;
		}else if(which_term==2){
			dipole_elements=pull_reduced_values_cc_vv(excitonic_momentum,dipole_elements,0);
			states1=spin_number_conduction_bands*number_conduction_bands*number_k_points_list;
		}else{
			dipole_elements=pull_reduced_values_cc_vv(excitonic_momentum,dipole_elements,1);
			states1=spin_number_valence_bands*number_valence_bands*number_k_points_list;
		}
	
		for(int i=0;i<states1;i++){
			for(int g=0;g<number_g_points_list;g++)
				cout<<dipole_elements(i,g)<<" ";
			cout<<endl;
		}
		for(int j=0;j<number_valence_plus_conduction*number_k_points_list;j++)
			cout<<energies.col(j)<<endl;
	};
};
tuple<mat,cx_mat> Dipole_Elements:: pull_values(vec excitonic_momentum){
	tuple<mat,cx_mat> ks_states_k_point; tuple<mat,cx_mat> ks_states_k_point_q;
	cx_mat ks_state; cx_mat ks_state_q; mat ks_energy; mat ks_energy_q;
	mat energies; energies.zeros(2,number_valence_bands*number_conduction_bands*number_k_points_list);

	cout<<"Calculating dipole elements... "<<endl;
	int htb_basis_dimension_2=htb_basis_dimension/2; int position;
	cout<<htb_basis_dimension<<" "<<number_conduction_bands<<" "<<number_valence_bands<<" "<<number_k_points_list<<" "<<number_g_points_list<<endl;
	
	cx_cube ks_state_right; ks_state_right.ones(htb_basis_dimension,number_valence_plus_conduction*number_k_points_list,number_g_points_list);
	////this is the heaviest but also the fastest solution (to use cx_cub for left state instead of cx_mat)
	cx_cube ks_state_left; ks_state_left.ones(htb_basis_dimension,number_valence_plus_conduction*number_k_points_list,number_g_points_list);
	cx_mat temporary_variable; temporary_variable.zeros(number_k_points_list,number_g_points_list);
	cx_mat temporary_variable_more; temporary_variable_more.zeros(htb_basis_dimension_2,number_g_points_list);

	cx_mat rho;

	cout<<"starting"<<endl;
	auto t1 = std::chrono::high_resolution_clock::now();
	if(spinorial_calculation==1){
		spin_number_valence_plus_conduction=2*number_valence_plus_conduction;
		rho.set_size(spin_number_valence_plus_conduction*number_valence_plus_conduction*number_k_points_list,number_g_points_list);
		for(int i=0;i<number_k_points_list;i++){
			cout<<"k point: "<<i<<endl;
			ks_states_k_point = hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i), number_valence_bands, number_conduction_bands);
			ks_states_k_point_q = hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i)-excitonic_momentum, number_valence_bands, number_conduction_bands);
			ks_state=get<1>(ks_states_k_point); ks_state_q=get<1>(ks_states_k_point_q);
			///adding exponential term e^{ikr}
			///e_{gl}k_{gm} -> l_{g(l,m)}
			for(int spin_channel=0;spin_channel<2;spin_channel++)
				for(int m=0;m<number_valence_plus_conduction;m++)
					for(int g=0;g<number_g_points_list;g++){
						ks_state_right.subcube(spin_channel*htb_basis_dimension_2,m*number_k_points_list+i,g,(spin_channel+1)*htb_basis_dimension_2-1,m*number_k_points_list+i,g)=
							exponential_factor.submat(spin_channel*htb_basis_dimension_2,g,(spin_channel+1)*htb_basis_dimension_2-1,g)%ks_state_q.submat(spin_channel*htb_basis_dimension_2,m,(spin_channel+1)*htb_basis_dimension_2-1,m);
						ks_state_left.subcube(spin_channel*htb_basis_dimension_2,m*number_k_points_list+i,g,(spin_channel+1)*htb_basis_dimension_2-1,m*number_k_points_list+i,g)=
							ks_state.submat(spin_channel*htb_basis_dimension_2,m,(spin_channel+1)*htb_basis_dimension_2-1,m);
					}
			ks_energy=get<0>(ks_states_k_point); ks_energy_q=get<0>(ks_states_k_point_q);
			for(int m=0;m<number_valence_bands;m++)
				for(int n=0;n<number_conduction_bands;n++)
					energies.col(m*number_conduction_bands*number_k_points_list+n*number_k_points_list+i)=ks_energy_q.col(m)-ks_energy.col(n+number_valence_bands);
		}
		///building screening function
		for(int spin_channel=0;spin_channel<2;spin_channel++)
			for(int m=0;m<number_valence_plus_conduction;m++)
				for(int n=0;n<number_valence_plus_conduction;n++){
					temporary_variable=sum(conj(ks_state_left.subcube(spin_channel*htb_basis_dimension_2,m*number_k_points_list,0,(spin_channel+1)*htb_basis_dimension_2-1,(m+1)*number_k_points_list-1,number_g_points_list-1))%
							ks_state_right.subcube(spin_channel*htb_basis_dimension_2,n*number_k_points_list,0,(spin_channel+1)*htb_basis_dimension_2-1,(n+1)*number_k_points_list-1,number_g_points_list-1),0);
					position=spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*number_k_points_list+m*number_valence_plus_conduction*number_k_points_list+n*number_k_points_list;
					rho.submat(position,0,position+number_k_points_list-1,number_g_points_list-1)=temporary_variable;
				}
	}else{
		spin_number_valence_plus_conduction=number_valence_plus_conduction;
		rho.set_size(spin_number_valence_plus_conduction*number_k_points_list,number_g_points_list);
		for(int i=0;i<number_k_points_list;i++){
			ks_states_k_point = hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i), number_valence_bands, number_conduction_bands);
			ks_states_k_point_q = hamiltonian_tb->pull_ks_states_subset(k_points_list.col(i)-excitonic_momentum, number_valence_bands, number_conduction_bands);
			ks_state=get<1>(ks_states_k_point); ks_state_q=get<1>(ks_states_k_point_q);
			///adding exponential term e^{ikr}
			///e_{gl}k_{gm} -> l_{g(l,m)}
			for(int m=0;m<number_valence_plus_conduction;m++)
				for(int g=0;g<number_g_points_list;g++){
					ks_state_right.subcube(0,m*number_k_points_list+i,g,htb_basis_dimension-1,m*number_k_points_list+i,g)=
						exponential_factor.submat(0,g,htb_basis_dimension-1,g)%ks_state_q.submat(0,m,htb_basis_dimension,m);
					ks_state_left.subcube(0,m*number_k_points_list+i,g,htb_basis_dimension,m*number_k_points_list+i,g)=
						ks_state.submat(0,m,htb_basis_dimension,m);
				}
			ks_energy=get<0>(ks_states_k_point); ks_energy_q=get<0>(ks_states_k_point_q);
			for(int m=0;m<number_valence_bands;m++)
				for(int n=0;n<number_conduction_bands;n++)
					energies.col(m*number_conduction_bands*number_k_points_list+n*number_k_points_list+i)=ks_energy_q.col(m)-ks_energy.col(n+number_valence_bands);
		}
		///building screening function
		for(int m=0;m<number_valence_plus_conduction;m++)
			for(int n=0;n<number_valence_plus_conduction;n++){
				temporary_variable=sum(conj(ks_state_left.subcube(0,m*number_k_points_list,0,htb_basis_dimension-1,(m+1)*number_k_points_list-1,number_g_points_list-1))%
						ks_state_right.subcube(0,n*number_k_points_list,0,htb_basis_dimension-1,(n+1)*number_k_points_list-1,number_g_points_list-1),0);
				position=m*number_valence_plus_conduction*number_k_points_list+n*number_k_points_list;
				rho.submat(position,0,position+number_k_points_list-1,number_g_points_list-1)=temporary_variable;
			}
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	cout<< std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    cout<< " milliseconds\n";
	
	return {energies,rho};
};
cx_mat Dipole_Elements:: pull_reduced_values_vc(vec excitonic_momentum, cx_mat rho){
	
	cx_mat rho_reduced;
	if(spinorial_calculation==1){
		spin_number_valence_bands=2*number_valence_bands;
		rho_reduced.set_size(spin_number_valence_bands*number_conduction_bands*number_k_points_list,number_g_points_list);
		//rho_reducde = rho(vc)
		for(int spin_channel=0;spin_channel<2;spin_channel++)
			for(int m=0;m<number_valence_bands;m++)
				rho_reduced.submat(spin_channel*number_valence_bands*number_conduction_bands*number_k_points_list+m*number_conduction_bands*number_k_points_list,0,spin_channel*number_valence_bands*number_conduction_bands*number_k_points_list+m*number_conduction_bands*number_k_points_list+number_conduction_bands*number_k_points_list-1,number_g_points_list-1)=
					rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*number_k_points_list+m*number_valence_plus_conduction*number_k_points_list+number_valence_bands*number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*number_k_points_list+m*number_valence_plus_conduction*number_k_points_list+number_valence_plus_conduction*number_k_points_list-1,number_g_points_list-1);
	}else{
		spin_number_valence_bands=number_valence_bands;
		rho_reduced.set_size(spin_number_valence_bands*number_conduction_bands*number_k_points_list,number_g_points_list);
		//rho_reducde = rho(vc)
		for(int m=0;m<number_valence_bands;m++)
			rho_reduced.submat(m*number_conduction_bands*number_k_points_list,0,m*number_conduction_bands*number_k_points_list+number_conduction_bands*number_k_points_list-1,number_g_points_list-1)=
				rho.submat(m*number_valence_plus_conduction*number_k_points_list+number_valence_bands*number_k_points_list,0,m*number_valence_plus_conduction*number_k_points_list+number_valence_plus_conduction*number_k_points_list-1,number_g_points_list-1);
	}
	return rho_reduced;
};
cx_mat Dipole_Elements:: pull_reduced_values_cc_vv(vec excitonic_momentum, cx_mat rho, int conduction_or_valence){

	cx_mat rho_reduced;
	if(spinorial_calculation==1){
		if(conduction_or_valence==0){
			spin_number_conduction_bands=2*number_conduction_bands;
			rho_reduced.set_size(spin_number_conduction_bands*number_conduction_bands*number_k_points_list,number_g_points_list);
			//rho_reducde = rho(cc)
			for(int spin_channel=0;spin_channel<2;spin_channel++)
				for(int m=0;m<number_conduction_bands;m++)
					rho_reduced.submat(spin_channel*number_conduction_bands*number_conduction_bands*number_k_points_list+m*number_conduction_bands*number_k_points_list,0,spin_channel*number_conduction_bands*number_conduction_bands*number_k_points_list+m*number_conduction_bands*number_k_points_list+number_conduction_bands*number_k_points_list-1,number_g_points_list-1)=
						rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*number_k_points_list+(number_valence_bands+m)*number_valence_plus_conduction*number_k_points_list+number_valence_bands*number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*number_k_points_list+(number_valence_bands+m)*number_valence_plus_conduction*number_k_points_list+number_valence_plus_conduction*number_k_points_list-1,number_g_points_list-1);
		}else{
			spin_number_valence_bands=2*number_valence_bands;
			rho_reduced.set_size(spin_number_valence_bands*number_valence_bands*number_k_points_list,number_g_points_list);
			//rho_reducde = rho(vv)
			for(int spin_channel=0;spin_channel<2;spin_channel++)
				for(int m=0;m<number_valence_bands;m++)
					rho_reduced.submat(spin_channel*number_valence_bands*number_valence_bands*number_k_points_list+m*number_valence_bands*number_k_points_list,0,spin_channel*number_valence_bands*number_valence_bands*number_k_points_list+m*number_valence_bands*number_k_points_list+number_valence_bands*number_k_points_list-1,number_g_points_list-1)=
						rho.submat(spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*number_k_points_list+m*number_valence_plus_conduction*number_k_points_list,0,spin_channel*number_valence_plus_conduction*number_valence_plus_conduction*number_k_points_list+m*number_valence_plus_conduction*number_k_points_list+number_valence_bands*number_k_points_list-1,number_g_points_list-1);
		}
	}else{
		if(conduction_or_valence==0){
			spin_number_conduction_bands=number_conduction_bands;
			rho_reduced.set_size(spin_number_conduction_bands*number_conduction_bands*number_k_points_list,number_g_points_list);
			//rho_reducde = rho(cc)
			for(int m=0;m<number_conduction_bands;m++)
				rho_reduced.submat(m*number_conduction_bands*number_k_points_list,0,m*number_conduction_bands*number_k_points_list+number_conduction_bands*number_k_points_list-1,number_g_points_list-1)=
					rho.submat((number_valence_bands+m)*number_valence_plus_conduction*number_k_points_list+number_valence_bands*number_k_points_list,0,(number_valence_bands+m)*number_valence_plus_conduction*number_k_points_list+number_valence_plus_conduction*number_k_points_list-1,number_g_points_list-1);
		}else{
			spin_number_valence_bands=number_valence_bands;
			rho_reduced.set_size(spin_number_valence_bands*number_valence_bands*number_k_points_list,number_g_points_list);
			//rho_reducde = rho(vv)
			for(int m=0;m<number_valence_bands;m++)
				rho_reduced.submat(m*number_valence_bands*number_k_points_list,0,m*number_valence_bands*number_k_points_list+number_valence_bands*number_k_points_list-1,number_g_points_list-1)=
					rho.submat(m*number_valence_plus_conduction*number_k_points_list,0,m*number_valence_plus_conduction*number_k_points_list+number_valence_bands*number_k_points_list-1,number_g_points_list-1);
		}
	}
	return rho_reduced;
};

class Dielectric_Function
{
private:
	int number_k_points_list;
	int number_g_points_list;
	int htb_basis_dimension;
	int number_valence_bands;
	int number_conduction_bands;
	vec excitonic_momentum;
	Dipole_Elements *dipole_elements;
	Coulomb_Potential *coulomb_potential;
	mat g_points_list;
	int spinorial_calculation;
	/// memory: to avoid recalculate rho many times, i.e. for different omega
	int saved_values;
	cx_mat rho_reduced;
	mat energies;
	vec old_excitonic_momentum;

public:
	Dielectric_Function(){
		number_k_points_list=0;
		number_g_points_list=0;
		htb_basis_dimension=0;
		number_conduction_bands=0;
		number_valence_bands=0;
		old_excitonic_momentum.zeros(3);
		saved_values=0;
	};
	void push_values(Dipole_Elements *dipole_elements_tmp,int number_k_points_list_tmp, int number_g_points_list_tmp, mat g_points_list_tmp, int number_valence_bands_tmp, int number_conduction_bands_tmp, Coulomb_Potential *coulomb_potential_tmp, int spinorial_calculation_tmp){
		number_conduction_bands=number_conduction_bands_tmp;
		number_valence_bands=number_valence_bands_tmp;
		number_k_points_list=number_k_points_list_tmp;
		number_g_points_list=number_g_points_list_tmp;
		dipole_elements=dipole_elements_tmp;
		coulomb_potential=coulomb_potential_tmp;
		g_points_list=g_points_list_tmp;
		spinorial_calculation=spinorial_calculation_tmp;
	};
	cx_mat pull_values(vec excitonic_momentum, cx_double omega, double eta);
	cx_mat pull_values_PPA(vec excitonic_momentum, cx_double omega, double eta, double PPA);
	void print(vec excitonic_momentum, cx_double omega, double eta, double PPA, int which_term);
	cx_vec pull_susceptability(vec excitonic_momentum, cx_vec omegas_path, int number_omegas_path, double eta, double limitq_0);
};
cx_mat Dielectric_Function::pull_values(vec excitonic_momentum, cx_double omega, double eta){
	cx_mat epsiloninv; epsiloninv.zeros(number_g_points_list,number_g_points_list);
	cx_double coulomb; vec excitonic_momentum_g(3);
	cx_double ieta; ieta.real(eta); ieta.imag(0.0);

	if((saved_values==0)||(accu(old_excitonic_momentum!=excitonic_momentum))){
		tuple<mat,cx_mat> energies_rho=dipole_elements->pull_values(excitonic_momentum);
		cx_mat rho=get<1>(energies_rho); 
		energies = get<0>(energies_rho);
		rho_reduced = dipole_elements->pull_reduced_values_vc(excitonic_momentum,rho);
		old_excitonic_momentum=excitonic_momentum;
		saved_values=1;
	}
	
	auto t1 = std::chrono::high_resolution_clock::now();
	cout<<"Calculating dielectric function..."<<endl;
	int position; cx_double energy;
	/// defining the denominator factors
	if(spinorial_calculation==1){
		cx_vec rho_reduced_single_column_modified(2*number_k_points_list*number_conduction_bands*number_valence_bands);
		cx_vec multiplicative_factor(2*number_k_points_list*number_conduction_bands*number_valence_bands);
		for(int spin_channel=0;spin_channel<2;spin_channel++)
			for(int i=0;i<number_k_points_list;i++)
				for(int c=0;c<number_conduction_bands;c++)
					for(int v=0;v<number_valence_bands;v++){
						energy=energies(spin_channel,v*number_conduction_bands*number_k_points_list+c*number_k_points_list+i);
						position=spin_channel*number_valence_bands*number_conduction_bands*number_k_points_list+v*number_conduction_bands*number_k_points_list+c*number_k_points_list+i;
						multiplicative_factor(position)=(1.0/(omega+energy+ieta))-(1.0/(omega-energy-ieta));
					}
		//cout<<"multiplicative factor calculated"<<endl;
		for(int i=0;i<number_g_points_list;i++){
			excitonic_momentum_g=excitonic_momentum+g_points_list.col(i);
			coulomb=coulomb_potential->pull(excitonic_momentum_g);
			rho_reduced_single_column_modified=rho_reduced.col(i)%multiplicative_factor;
			for(int j=0;j<number_g_points_list;j++)
				epsiloninv(i,j)=coulomb*accu(conj(rho_reduced.col(j))%rho_reduced_single_column_modified)/(8*pow(pigreco,3));
		}
	}else{
		cx_vec rho_reduced_single_column_modified(number_k_points_list*number_conduction_bands*number_valence_bands);
		cx_vec multiplicative_factor(number_k_points_list*number_conduction_bands*number_valence_bands);
		for(int i=0;i<number_k_points_list;i++)
			for(int c=0;c<number_conduction_bands;c++)
				for(int v=0;v<number_valence_bands;v++){
					energy=energies(0,v*number_conduction_bands*number_k_points_list+c*number_k_points_list+i);
					position=v*number_conduction_bands*number_k_points_list+c*number_k_points_list+i;
					multiplicative_factor(position)=(1.0/(omega+energy+ieta))-(1.0/(omega-energy-ieta));
				}
		for(int i=0;i<number_g_points_list;i++){
			excitonic_momentum_g=excitonic_momentum+g_points_list.col(i);
			coulomb=coulomb_potential->pull(excitonic_momentum_g);
			rho_reduced_single_column_modified=rho_reduced.col(i)%multiplicative_factor;
			for(int j=0;j<number_g_points_list;j++)
				epsiloninv(i,j)=coulomb*accu(conj(rho_reduced.col(j))%rho_reduced_single_column_modified)/(4*pow(pigreco,3));
		}
	}
	for(int i=0;i<number_g_points_list;i++)
		epsiloninv(i,i).real(real(epsiloninv(i,i))+1.0);

	auto t2 = std::chrono::high_resolution_clock::now();
	cout<< std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    cout<< " milliseconds\n";
	return epsiloninv;
};
cx_mat Dielectric_Function:: pull_values_PPA(vec excitonic_momentum, cx_double omega, double eta, double PPA){
	cx_double omega_PPA; omega_PPA.imag(PPA); omega_PPA.real(0.0);
	cx_double omega_0; omega_0.real(0.0); omega_0.imag(0.0);
	cx_double ieta; ieta.real(0.0); ieta.imag(eta);
	
	cx_mat epsiloninv_0=pull_values(excitonic_momentum,omega_0,eta);
	cx_mat epsiloninv_PPA=pull_values(excitonic_momentum,omega_PPA,eta);
	
	cx_mat rgg(number_g_points_list,number_g_points_list);
	cx_mat ogg(number_g_points_list,number_g_points_list);

	ogg=PPA*sqrt(epsiloninv_PPA/(epsiloninv_0-epsiloninv_PPA));
	rgg=(epsiloninv_0%ogg)/2;

	cx_mat epsilon_app(number_g_points_list,number_g_points_list,fill::zeros);
	for(int i=0;i<number_g_points_list;i++)
		for(int j=0;j<number_g_points_list;j++)
			epsilon_app(i,j)=rgg(i,j)*(1.0/(omega-ogg(i,j)+ieta)-1.0/(omega+ogg(i,j)-ieta));

	for(int i=0;i<number_g_points_list;i++)
		epsilon_app(i,i).real(real(epsilon_app(i,i))+1.0);

	return epsilon_app;
};
void Dielectric_Function::print(vec excitonic_momentum, cx_double omega, double eta, double PPA, int which_term){
	cx_mat dielectric_function;
	
	if(which_term==0)
		dielectric_function=pull_values(excitonic_momentum,omega,eta);
	else
		dielectric_function=pull_values_PPA(excitonic_momentum,omega,eta,PPA);
	
	for(int i=0;i<number_g_points_list;i++){
		for(int j=0;j<number_g_points_list;j++)
			cout<<dielectric_function(i,j)<<" ";
		cout<<endl;
	}
};
cx_vec Dielectric_Function::pull_susceptability(vec excitonic_momentum,cx_vec omegas_path,int number_omegas_path,double eta,double limitq_0){
	cx_mat dielectric_susceptability;
	cx_vec dielectric_susceptability_along_omegas_path;
	dielectric_susceptability_along_omegas_path.zeros(number_omegas_path); 
	vec k_point_0; k_point_0.zeros(3); k_point_0=k_point_0+limitq_0;

	for(int i=0;i<number_omegas_path;i++){
		cout<<"Omega: "<<omegas_path(i)<<endl;
		dielectric_susceptability=pull_values(k_point_0,omegas_path(i),eta);
		dielectric_susceptability_along_omegas_path(i)=dielectric_susceptability(0,0);
	}
	return dielectric_susceptability_along_omegas_path;
};

/// Excitonic_Hamiltonian class
class Excitonic_Hamiltonian
{
private:
	int spinorial_calculation;
	int number_valence_bands;
	int number_conduction_bands;
	int number_valence_plus_conduction;
	int number_valence_times_conduction;
	int dimension_bse_hamiltonian;
	int spin_dimension_bse_hamiltonian;
	int spin_number_valence_plus_conduction;
	int spin_number_valence_times_conduction;
	int htb_basis_dimension;
	int bse_basis_dimension;	
	int number_k_points_list;
	int number_g_points_list;
	mat k_points_list;
	mat g_points_list;
	mat exciton;
	mat exciton_spin;
	Coulomb_Potential *coulomb_potential;
	Dielectric_Function *dielectric_function;
	Hamiltonian_TB *hamiltonian_tb;
	Dipole_Elements *dipole_elements;
	cx_mat excitonic_hamiltonian;
	int adding_screening;

public:
	Excitonic_Hamiltonian()
	{
		spinorial_calculation = 0;
		number_k_points_list = 0;
		number_conduction_bands = 0;
		number_valence_bands = 0;
		number_valence_plus_conduction = 0;
		spin_number_valence_plus_conduction = 0;
		dimension_bse_hamiltonian = 0;
		htb_basis_dimension = 0;
		bse_basis_dimension = 0;
	}
	/// be carefull: do not try to build the BSE matrix with more bands than those given by the hamiltonian!!!
	/// there is a check at the TB hamiltonian level but not here...
	void push_values(int number_valence_bands_tmp, int number_conduction_bands_tmp, Coulomb_Potential *coulomb_potential_tmp, Dielectric_Function *dielectric_function_tmp, Hamiltonian_TB *hamiltonian_tb_tmp, Dipole_Elements *dipole_elements_tmp, mat k_points_list_tmp, int number_k_points_list_tmp, mat g_points_list_tmp,int number_g_points_list_tmp, int spinorial_calculation_tmp, int adding_screening_tmp)
	{
		spinorial_calculation = spinorial_calculation_tmp;
		number_k_points_list = number_k_points_list_tmp;
		number_conduction_bands = number_conduction_bands_tmp;
		number_valence_bands = number_valence_bands_tmp;
		number_valence_plus_conduction = number_conduction_bands + number_valence_bands;
		number_valence_times_conduction = number_conduction_bands * number_valence_bands;
		dimension_bse_hamiltonian = number_k_points_list * number_conduction_bands * number_valence_bands;
		k_points_list = k_points_list_tmp;
		g_points_list = g_points_list_tmp;

		number_g_points_list = number_g_points_list_tmp;
		hamiltonian_tb = hamiltonian_tb_tmp;
		htb_basis_dimension = hamiltonian_tb->pull_htb_basis_dimension();
		bse_basis_dimension = htb_basis_dimension * 2;

		coulomb_potential = coulomb_potential_tmp;
		adding_screening=adding_screening_tmp;
		dielectric_function=dielectric_function_tmp;

		dipole_elements=dipole_elements_tmp;

		exciton.set_size(2, number_valence_times_conduction);
		int e = 0;
		for (int v = 0; v < number_valence_bands; v++)
			for (int c = 0; c < number_conduction_bands; c++)
			{
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

		if (spinorial_calculation == 1)
		{
			spin_dimension_bse_hamiltonian = dimension_bse_hamiltonian * 4;
			spin_number_valence_plus_conduction = number_valence_plus_conduction * 2;
			spin_number_valence_times_conduction = number_valence_times_conduction * 4;
		}
		else
		{
			spin_dimension_bse_hamiltonian = dimension_bse_hamiltonian;
			spin_number_valence_plus_conduction = number_valence_plus_conduction;
			spin_number_valence_times_conduction = number_valence_times_conduction;
		}
	}
	cx_mat pull_excitonic_hamiltonian(vec excitonic_momentum, double epsilon, double eta);
	tuple<vec, cx_mat> pull_eigenstates_through_cholesky(cx_mat excitonic_hamiltonian);
	tuple<cx_vec, cx_mat> pull_eigenstates_through_usualway(cx_mat excitonic_hamiltonian);
	cx_cube pull_excitonic_oscillator_force(cx_mat excitonic_eigenstates, cx_mat dipoles);
	void pull_dielectric_tensor_bse(vec excitonic_momentum, double eta, double epsilon, ofstream *file_diel, double scissor_operator, double energy_step, double max_energy);
	void print(vec excitonic_momentum, double epsilon, double eta)
	{		
		cout<<"BSE hamiltoian..."<<endl;
		cx_mat hamiltonian = pull_excitonic_hamiltonian(excitonic_momentum, epsilon, eta);
		for(int i=0;i<spin_dimension_bse_hamiltonian;i++){
			for(int j=0;j<spin_dimension_bse_hamiltonian;j++)
				printf("(%2.6f|%2.6f)",hamiltonian(i,j).real(),hamiltonian(i,j).imag());
			cout<<endl;
		}
		//cout<<"Dipoles..."<<endl;
		//cx_mat dipoles;
		//for(int i=0;i<number_k_points_list;i++){
		//	dipoles=hamiltonian_tb->extract_usual_dipoles(k_points_list.col(i),number_valence_bands,number_conduction_bands,eta);
		//	for (int xyz = 0;xyz<3;xyz++)
		//		for (int q = 0;q<2*number_valence_times_conduction;q++)
		//			printf("(%.4f+i%.4f)| ",real(dipoles(q,xyz)),imag(dipoles(q,xyz)));
		//}
		cout<<"Eigenvalues..."<<endl;
		tuple<cx_vec, cx_mat> eigenvalues_and_eigenstates;
		eigenvalues_and_eigenstates=pull_eigenstates_through_usualway(hamiltonian);
		cx_vec eigenvalues=get<0>(eigenvalues_and_eigenstates);
		cx_mat eigenstates=get<1>(eigenvalues_and_eigenstates);
		for (int i=0;i<spin_dimension_bse_hamiltonian;i++)
			if(real(eigenvalues(i))>=0)
				printf("%2.6f\n", real(eigenvalues(i)));
	};
	cx_cube pull_excitonic_oscillator_force(cx_mat excitonic_eigenstates, double eta);
};
cx_mat Excitonic_Hamiltonian::pull_excitonic_hamiltonian(vec excitonic_momentum, double epsilon, double eta)
{
	/// saving memory for the BSE matrix (kernel)
	cx_mat excitonic_hamiltonian;
	excitonic_hamiltonian.set_size(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian);

	/// building the BSE matrix
	/// calculating the potentianl before the BSE hamiltonian building
	cx_cube v_coulomb_diff(number_g_points_list,number_g_points_list,number_k_points_list*number_k_points_list, fill::zeros);
	cx_vec v_coloumb_q(number_g_points_list);
	vec excitonic_momentum_0; excitonic_momentum_0.zeros(3);
	vec k_point_diff_g; k_point_diff_g.set_size(3); vec k_point_q_g; k_point_q_g.set_size(3);
	tuple<mat,cx_mat> energies_rho_0=dipole_elements->pull_values(excitonic_momentum_0);
	mat energies_0=get<0>(energies_rho_0); cx_mat rho_0=get<1>(energies_rho_0);
	tuple<mat,cx_mat> energies_rho_q=dipole_elements->pull_values(excitonic_momentum);
	mat energies_q=get<0>(energies_rho_q); cx_mat rho_q=get<1>(energies_rho_q);

	///calculating screening and diagonal terms of the BSE hamiltonian
	cx_mat epsilon_inv_static;
	if(adding_screening==1){
		cx_double omega_0; omega_0.real(0.0); omega_0.imag(0.0);
		epsilon_inv_static=dielectric_function->pull_values(excitonic_momentum,omega_0,eta);
	}else{
		epsilon_inv_static.set_size(number_g_points_list,number_g_points_list);
		epsilon_inv_static.diag(1.0);
	}
	// calculating the generalized potential (the screened one and the unscreened-one)
	for (int k = 0; k < number_g_points_list; k++){
		for (int s = 0; s < number_g_points_list; s++)
			for (int i = 0; i < number_k_points_list; i++)
				for (int j = 0; j < number_k_points_list; j++){
					k_point_diff_g = k_points_list.col(i) - k_points_list.col(j) + g_points_list.col(k);
					v_coulomb_diff(k,s,i*number_k_points_list+j) = epsilon_inv_static(k,s)*coulomb_potential->pull(k_point_diff_g);
				}
		k_point_q_g=excitonic_momentum+g_points_list.col(k);
		v_coloumb_q(k) = coulomb_potential ->pull(k_point_q_g);
	}

	cout <<"Building dipole elements for BSE hamiltonian..."<< endl;
	cx_mat rho_cc=dipole_elements->pull_reduced_values_cc_vv(excitonic_momentum,rho_q,0);
	cx_mat rho_vv=dipole_elements->pull_reduced_values_cc_vv(excitonic_momentum,rho_q,1);
	cx_mat rho_vc_0=dipole_elements->pull_reduced_values_vc(excitonic_momentum_0,rho_0);

	cx_mat w_matrix(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian);
	cx_mat v_matrix(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian);

	const auto start = std::chrono::system_clock::now();
	cout <<"Building BSE hamiltonian..."<< endl;

	if (spinorial_calculation == 1)
	{
		int spin_c1; int spin_v1; int spin_c2; int spin_v2;
		int row; int column;
		cx_vec temporary_matrix1(number_g_points_list);
		cx_vec temporary_matrix2(number_g_points_list);
		for(int i=0;i<number_k_points_list;i++)
			for(int v1=0;v1<number_valence_bands;v1++)
				for(int c1=0;c1<number_conduction_bands;c1++)
					for(int spin_channel1=0;spin_channel1<4;spin_channel1++){
						spin_v1=exciton_spin(0,spin_channel1);
						spin_c1=exciton_spin(1,spin_channel1);
						for(int j=0;j<number_k_points_list;j++)
							for(int v2=0;v2<number_valence_bands;v2++)
								for(int c2=0;c2<number_conduction_bands;c2++)
									for(int spin_channel2=0;spin_channel2<4;spin_channel2++){
										spin_v2=exciton_spin(0,spin_channel2);
										spin_c2=exciton_spin(1,spin_channel2);
										row=spin_v1*2*number_conduction_bands*number_valence_bands*number_k_points_list+spin_c1*number_conduction_bands*number_valence_bands*number_k_points_list+v1*number_conduction_bands*number_k_points_list+c1*number_k_points_list+i;
										column=spin_v2*2*number_conduction_bands*number_valence_bands*number_k_points_list+spin_c2*number_conduction_bands*number_valence_bands*number_k_points_list+v2*number_conduction_bands*number_k_points_list+c2*number_k_points_list+j;
										temporary_matrix1=(v_coulomb_diff.slice(i*number_k_points_list+j))*(rho_cc.row(spin_c1*number_conduction_bands*number_conduction_bands*number_k_points_list+c1*number_conduction_bands*number_k_points_list+c2*number_k_points_list+i).t());
										w_matrix(row,column)=accu((temporary_matrix1.t())%conj(rho_vv.row(spin_v1*number_valence_bands*number_valence_bands*number_k_points_list+v1*number_valence_bands*number_k_points_list+v2*number_k_points_list+i)));				
										temporary_matrix2=v_coloumb_q%((rho_vc_0.row(spin_v1*number_valence_bands*number_conduction_bands*number_k_points_list+v1*number_conduction_bands*number_k_points_list+c1*number_k_points_list+i)).t());
										v_matrix(row,column)=accu(temporary_matrix2%((rho_vc_0.row(spin_v2*number_valence_bands*number_conduction_bands*number_k_points_list+v2*number_conduction_bands*number_k_points_list+c2*number_k_points_list+j)).t()));
									}
					}
		cout<<"end calculation coupling elements"<<endl;
		excitonic_hamiltonian=(w_matrix+v_matrix)/coulomb_potential->pull_cell_volume();

		/// adding the diagonal part to the BSE hamiltonian
		for(int i=0;i<number_k_points_list;i++)
			for(int v1=0;v1<number_valence_bands;v1++)
				for(int c1=0;c1<number_conduction_bands;c1++)
					for(int spin_channel1=0;spin_channel1<4;spin_channel1++){
						spin_v1=exciton_spin(0,spin_channel1);
						row=spin_v1*2*number_conduction_bands*number_valence_bands*number_k_points_list+spin_v1*number_conduction_bands*number_valence_bands*number_k_points_list+v1*number_conduction_bands*number_k_points_list+c1*number_k_points_list+i;
						excitonic_hamiltonian(row,row)=energies_q(spin_v1,v1*number_conduction_bands*number_k_points_list+c1*number_k_points_list+i);
					}
		cout<<"end calculation diagonal elements"<<endl;
	}
	else
	{
		int row; int column;
		cx_vec temporary_matrix1(number_g_points_list); 
		cx_vec temporary_matrix2(number_g_points_list);
		for(int i=0;i<number_k_points_list;i++)
			for(int v1=0;v1<number_valence_bands;v1++)
				for(int c1=0;c1<number_conduction_bands;c1++)
						for(int j=0;j<number_k_points_list;j++)
							for(int v2=0;v2<number_valence_bands;v2++)
								for(int c2=0;c2<number_conduction_bands;c2++){
									row=v1*number_conduction_bands*number_k_points_list+c1*number_k_points_list+i;
									column=v2*number_conduction_bands*number_k_points_list+c2*number_k_points_list+j;
									temporary_matrix1=(v_coulomb_diff.slice(i*number_k_points_list+j))*(rho_cc.row(c1*number_conduction_bands*number_k_points_list+c2*number_k_points_list+i).t());
									w_matrix(row,column)=accu((temporary_matrix1.t())%rho_vv.row(v1*number_valence_bands*number_k_points_list+v2*number_k_points_list+i));				
									temporary_matrix2=(v_coloumb_q%(rho_vc_0.row(v1*number_conduction_bands*number_k_points_list+c1*number_k_points_list+i).t()));
									v_matrix(row,column)=accu(temporary_matrix2%(rho_vc_0.row(v2*number_conduction_bands*number_k_points_list+c2*number_k_points_list+i).t()));
								}

		cout<<"end calculation coupling elements"<<endl;
		excitonic_hamiltonian=(w_matrix+v_matrix)/coulomb_potential->pull_cell_volume();

		/// adding the diagonal part to the BSE hamiltonian
		for(int i=0;i<number_k_points_list;i++)
			for(int v1=0;v1<number_valence_bands;v1++)
				for(int c1=0;c1<number_conduction_bands;c1++){
					row=v1*number_conduction_bands*number_k_points_list+c1*2*number_k_points_list+i;
					excitonic_hamiltonian(row,row)=energies_q(0,v1*number_conduction_bands*number_k_points_list+c1*number_k_points_list+i);
				}
		cout<<"end calculation diagonal elements"<<endl;
	}

	cout << "Building BSE hamiltonian finished..." << endl;
	const auto end = std::chrono::system_clock::now();
	const auto duration = std::chrono::duration<double>{end - start};
	cout << "Timing needed " << duration.count() << '\n';
	return excitonic_hamiltonian;
};
/// usual diagonalization routine
tuple<cx_vec, cx_mat> Excitonic_Hamiltonian::pull_eigenstates_through_usualway(cx_mat excitonic_hamiltonian)
{
	/// diagonalizing the BSE matrix
	/// M_{(bz_number_k_points x number_valence_bands x number_conduction_bands)x(bz_number_k_points x number_valence_bands x number_conduction_bands)}
	if (spinorial_calculation == 1)
	{
		int dimension_bse_hamiltonian_3=3*dimension_bse_hamiltonian;
		int dimension_bse_hamiltonian_4=4*dimension_bse_hamiltonian;

		cx_vec eigenvalues_1; cx_mat eigenvectors_1; cx_vec exc_eigenvalues_1;
		cx_vec eigenvalues_0; cx_mat eigenvectors_0; cx_vec exc_eigenvalues_0;
		cx_mat excitonic_hamiltonian_1; cx_mat excitonic_hamiltonian_0;
		excitonic_hamiltonian_1.set_size(dimension_bse_hamiltonian_3,dimension_bse_hamiltonian_3);
		excitonic_hamiltonian_0.set_size(dimension_bse_hamiltonian,dimension_bse_hamiltonian);

		///the two spin channels are not coupling to each other because of the missing spin-orbit coupling
		///considering the Clebsch-Gorddan coefficients we obtain the transformation matrix between the two spin representations
		cx_mat transformation_matrix; transformation_matrix.zeros(4,4);
		transformation_matrix(0,2)=1.0; transformation_matrix(3,0)=1.0;
		transformation_matrix(1,1)=1.0/sqrt(2); transformation_matrix(1,3)=-1.0/sqrt(2);
		transformation_matrix(2,1)=1.0/sqrt(2); transformation_matrix(2,3)=1.0/sqrt(2);
		int spin_c1; int spin_c2; int spin_v1; int spin_v2;
		cx_mat temporary_matrix(dimension_bse_hamiltonian,dimension_bse_hamiltonian);
		cx_mat transformed_excitonic_hamiltonian;
		transformed_excitonic_hamiltonian.zeros(dimension_bse_hamiltonian_4,dimension_bse_hamiltonian_4);
		for(int spin_channel1=0;spin_channel1<4;spin_channel1++)
			for(int spin_channel2=0;spin_channel2<4;spin_channel2++){
				temporary_matrix.zeros(dimension_bse_hamiltonian,dimension_bse_hamiltonian);
				for(int spin_channel3=0;spin_channel3<4;spin_channel3++)
					for(int spin_channel4=0;spin_channel4<4;spin_channel4++){
						spin_v1=exciton_spin(0,spin_channel3);
						spin_c1=exciton_spin(1,spin_channel3);
						spin_v2=exciton_spin(0,spin_channel4);
						spin_c2=exciton_spin(1,spin_channel4);
						temporary_matrix=temporary_matrix+(transformation_matrix(spin_channel1,spin_channel3))*(excitonic_hamiltonian.submat(spin_v1*2*dimension_bse_hamiltonian+spin_c1*dimension_bse_hamiltonian,spin_v2*2*dimension_bse_hamiltonian+spin_c2*dimension_bse_hamiltonian,spin_v1*2*dimension_bse_hamiltonian+(spin_c1+1)*dimension_bse_hamiltonian-1,spin_v2*2*dimension_bse_hamiltonian+(spin_c2+1)*dimension_bse_hamiltonian-1))*transformation_matrix(spin_channel4,spin_channel2);
					}
					
				transformed_excitonic_hamiltonian.submat(spin_channel1*dimension_bse_hamiltonian,spin_channel2*dimension_bse_hamiltonian,(spin_channel1+1)*dimension_bse_hamiltonian-1,(spin_channel2+1)*dimension_bse_hamiltonian-1)=temporary_matrix;
			}

		excitonic_hamiltonian_1=transformed_excitonic_hamiltonian.submat(0,0,dimension_bse_hamiltonian_3-1,dimension_bse_hamiltonian_3-1);
		excitonic_hamiltonian_0=transformed_excitonic_hamiltonian.submat(dimension_bse_hamiltonian_3,dimension_bse_hamiltonian_3,dimension_bse_hamiltonian_4-1,dimension_bse_hamiltonian_4-1);

		///diagonalizing the two spin channels: singlet and triplet
		eig_gen(eigenvalues_1, eigenvectors_1, excitonic_hamiltonian_1);
		eig_gen(eigenvalues_0, eigenvectors_0, excitonic_hamiltonian_0);

		cx_mat exc_eigenvectors; cx_vec exc_eigenvalues;
		exc_eigenvectors.zeros(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian);
		exc_eigenvalues.zeros(spin_dimension_bse_hamiltonian);

		///ordering the eigenvalues and saving them in a single matrix exc_eigenvalues
		exc_eigenvalues_1=sort(eigenvalues_1); exc_eigenvalues_0=sort(eigenvalues_0);
		uvec ordering_1=sort_index(eigenvalues_1); uvec ordering_0=sort_index(eigenvalues_0);
		/// normalizing and ordering eigenvectors: triplet and then singlet; and saving them in a single matrix exc_eigenvectors
		double modulus;
		for(int i=0;i<dimension_bse_hamiltonian_3;i++)
		{
			modulus=norm(eigenvectors_1.col(ordering_1(i)),2);
			for(int s=0;s<dimension_bse_hamiltonian_3;s++)
				exc_eigenvectors(s,i)=eigenvectors_1(s,ordering_1(i))/modulus; 
			exc_eigenvalues(i) = exc_eigenvalues_1(i);
		}
		for (int i=0;i<dimension_bse_hamiltonian;i++)
		{
			modulus=norm(eigenvectors_0.col(ordering_0(i)),2);
			for(int s=0;s<dimension_bse_hamiltonian;s++)
				exc_eigenvectors(s+dimension_bse_hamiltonian_3,i+dimension_bse_hamiltonian_3)=eigenvectors_0(s,ordering_0(i))/modulus; 
			exc_eigenvalues(i+dimension_bse_hamiltonian_3) = exc_eigenvalues_0(i);
		}
		return {exc_eigenvalues, exc_eigenvectors};
	}
	else
	{
		cx_vec eigenvalues; cx_mat eigenvectors;
		eig_gen(eigenvalues,eigenvectors,excitonic_hamiltonian);
		cx_vec exc_eigenvalues; exc_eigenvalues.set_size(spin_dimension_bse_hamiltonian);
		cx_mat exc_eigenvectors; exc_eigenvectors.set_size(spin_dimension_bse_hamiltonian, spin_dimension_bse_hamiltonian);

		exc_eigenvalues = sort(eigenvalues);
		uvec ordering = sort_index(eigenvalues);
		/// normalizing and ordering eigenvectors
		double modulus;
		for (int i = 0; i < spin_dimension_bse_hamiltonian; i++)
		{
			modulus=norm(eigenvectors.col(ordering(i)),2);
			exc_eigenvectors.col(i)=eigenvectors.col(ordering(i))/modulus; 
			exc_eigenvalues(i) = exc_eigenvalues(i);
		}
		return {exc_eigenvalues, exc_eigenvectors};
	}
};
/// FASTESTS diagonalization routine
/// Structure preserving parallel algorithms for solving the Bethe–Salpeter eigenvalue problem Meiyue Shao, Felipe H. da Jornada, Chao Yang, Jack Deslippe, Steven G. Louie
tuple<vec, cx_mat> Excitonic_Hamiltonian::pull_eigenstates_through_cholesky(cx_mat excitonic_hamiltonian)
{
	/// diagonalizing the BSE matrix M_{(bz_number_k_points x number_valence_bands x number_conduction_bands)x(bz_number_k_points x number_valence_bands x number_conduction_bands)}
	int dimension_bse_hamiltonian_2 = spin_dimension_bse_hamiltonian / 2;
	cx_mat A;
	cx_mat B;

	A.set_size(dimension_bse_hamiltonian_2, dimension_bse_hamiltonian_2);
	B.set_size(dimension_bse_hamiltonian_2, dimension_bse_hamiltonian_2);
	for (int q = 0; q < dimension_bse_hamiltonian; q++)
		for (int s = 0; s < dimension_bse_hamiltonian; s++)
		{
			if ((q < dimension_bse_hamiltonian_2) && (s < dimension_bse_hamiltonian_2))
				A(q, s) = excitonic_hamiltonian(q, s);
			if ((q < dimension_bse_hamiltonian_2) && (s >= dimension_bse_hamiltonian_2))
				B(q, s - dimension_bse_hamiltonian_2) = excitonic_hamiltonian(q, s);
		}

	cx_mat ABdiff = A - B;
	cx_mat ABsum = A + B;
	cx_mat M;
	M.set_size(spin_dimension_bse_hamiltonian, spin_dimension_bse_hamiltonian);

	for (int q = 0; q < spin_dimension_bse_hamiltonian; q++)
		for (int s = 0; s < dimension_bse_hamiltonian; s++)
		{
			if ((q < dimension_bse_hamiltonian_2) && (s < dimension_bse_hamiltonian_2))
				M(q, s) = real(ABsum(q, s));
			else if ((q < dimension_bse_hamiltonian_2) && (s >= dimension_bse_hamiltonian_2))
				M(q, s) = imag(ABdiff(q, s - dimension_bse_hamiltonian_2));
			else if ((q >= dimension_bse_hamiltonian_2) && (s < dimension_bse_hamiltonian_2))
				M(q, s) = -imag(ABsum(q - dimension_bse_hamiltonian_2, s));
			else
				M(q, s) = real(ABdiff(q - dimension_bse_hamiltonian_2, q - dimension_bse_hamiltonian_2));
		}

	/// compute the Cholesky factorization
	cx_mat L = chol(M);
	/// construct W
	cx_mat J;
	J.zeros(spin_dimension_bse_hamiltonian, spin_dimension_bse_hamiltonian);
	for (int q = 0; q < dimension_bse_hamiltonian_2; q++)
		for (int s = 0; s < dimension_bse_hamiltonian_2; s++)
		{
			J(q, s + dimension_bse_hamiltonian_2) = 1.00;
			J(q + dimension_bse_hamiltonian_2, s) = -1.00;
		}
	cx_mat W = M * J * M;

	vec eigenvalues;
	cx_mat eigenvectors;
	eig_sym(eigenvalues, eigenvectors, W);

	vec exc_eigenvalues;
	exc_eigenvalues.set_size(spin_dimension_bse_hamiltonian);
	exc_eigenvalues = sort(eigenvalues);
	/// normalizing and ordering eigenvectors
	mat ordering(spin_dimension_bse_hamiltonian, 2, fill::zeros);
	for (int i = 0; i < spin_dimension_bse_hamiltonian; i++)
		if (ordering(i, 1) != 1)
			for (int j = 0; j < spin_dimension_bse_hamiltonian; j++)
				if (exc_eigenvalues(i) == eigenvalues(j))
				{
					ordering(i, 0) = j;
					ordering(i, 1) = 1;
				}
	cx_mat exc_eigenvectors;
	exc_eigenvectors.set_size(spin_dimension_bse_hamiltonian, spin_dimension_bse_hamiltonian);
	cx_double temporary_norm;
	for (int i = 0; i < spin_dimension_bse_hamiltonian; i++)
	{
		temporary_norm = accu(eigenvectors.col(ordering(i, 0)) % eigenvectors.col(ordering(i, 0)));
		exc_eigenvectors.col(ordering(i, 0)) = eigenvectors.col(ordering(i, 0)) / temporary_norm;
	}
	return {exc_eigenvalues, exc_eigenvectors};
};
cx_cube Excitonic_Hamiltonian::pull_excitonic_oscillator_force(cx_mat excitonic_eigenstates, double eta){
	
	cx_mat dipoles; dipoles.zeros(spin_dimension_bse_hamiltonian, 3);
	cx_mat dipoles_single_k_point;
	vec k_point(3);
	if(spinorial_calculation==1){
		for(int i=0;i<number_k_points_list;i++){
			k_point=k_points_list.col(i);
			dipoles_single_k_point=hamiltonian_tb->extract_usual_dipoles(k_point,number_valence_bands,number_conduction_bands,eta);
			for(int xyz=0;xyz<3;xyz++)
				for(int spin_channel=0;spin_channel<2;spin_channel++)
					dipoles.submat(3*spin_channel*dimension_bse_hamiltonian+i*number_valence_times_conduction,xyz,3*spin_channel*dimension_bse_hamiltonian+(i+1)*number_valence_times_conduction-1,xyz)=
						dipoles_single_k_point.submat(spin_channel*number_valence_times_conduction,xyz,(spin_channel+1)*number_valence_times_conduction-1,xyz);
		}
	}else{
		for(int i=0;i<number_k_points_list;i++){
			k_point=k_points_list.col(i);
			dipoles_single_k_point=hamiltonian_tb->extract_usual_dipoles(k_point,number_valence_bands,number_conduction_bands,eta);
			for(int xyz=0;xyz<3;xyz++)
				dipoles.submat(i*number_valence_times_conduction,xyz,(i+1)*number_valence_times_conduction-1,xyz)=
					dipoles_single_k_point.col(xyz);
		}
	}
	
	//dipoles=function_separating_spin_channels_dipoles(dipoles,spin_dimension_bse_hamiltonian,dimension_bse_hamiltonian,number_valence_times_conduction,number_k_points_list);
	
	cx_cube oscillator_force(3, 3, spin_dimension_bse_hamiltonian, fill::zeros);
	//cout<<"TEST PRINT OSCILLATOR FORCE"<<endl;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			for (int q = 0; q < spin_dimension_bse_hamiltonian; q++)
				oscillator_force(i, j, q)=(accu(excitonic_eigenstates.col(q)%dipoles.col(i)))*conj(accu(excitonic_eigenstates.col(q)%dipoles.col(j)));
	
	return oscillator_force;
};
void Excitonic_Hamiltonian:: pull_dielectric_tensor_bse(vec excitonic_momentum, double eta, double epsilon, ofstream *file_diel, double scissor_operator, double energy_step, double max_energy)
{
	cout << "Calculation of the dielectric tensor..." << endl;
	cx_mat exc_hamiltonian(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian);
	exc_hamiltonian=pull_excitonic_hamiltonian(excitonic_momentum, epsilon, eta);
	cx_vec exc_eigenvalues(spin_dimension_bse_hamiltonian); 
	cx_mat exc_eigenstates(spin_dimension_bse_hamiltonian,spin_dimension_bse_hamiltonian); 
	tuple<cx_vec,cx_mat> eigenvalues_and_eigenstates;
	eigenvalues_and_eigenstates = pull_eigenstates_through_usualway(exc_hamiltonian);
	exc_eigenvalues=get<0>(eigenvalues_and_eigenstates); exc_eigenstates=get<1>(eigenvalues_and_eigenstates);

	cx_cube exc_oscillator_force(3, 3, spin_dimension_bse_hamiltonian);
	exc_oscillator_force=pull_excitonic_oscillator_force(exc_eigenstates,eta);

	double factor=1000*(pow(const_electron_charge, 2)/(const_vacuum_dielectric_constant*coulomb_potential->pull_cell_volume()*number_k_points_list));
	
	int number_of_energy_steps=max_energy/energy_step;
	vec omega(number_of_energy_steps);
	for(int i=0;i<number_of_energy_steps;i++)
		omega(i)=double(i*max_energy/(number_of_energy_steps-1));

	cx_cube dielectric_tensor_bse(3, 3, number_of_energy_steps,fill::zeros);
	cx_double variable_tmp_1; double variable_tmp_2;

	cout<<"proper dielectric tensor"<<endl;
	#pragma omp parallel for
	for (int r = 0; r < number_of_energy_steps; r++){
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++){
				for (int q = 0; q < spin_dimension_bse_hamiltonian; q++)
				{
					variable_tmp_1 = exc_oscillator_force(i, j, q)/(pow(hbar*omega(r) - exc_eigenvalues(q), 2) + pow(eta, 2));
					variable_tmp_2 = hbar*omega(r)-real(exc_eigenvalues(q));
					dielectric_tensor_bse(i, j, r).real(real(dielectric_tensor_bse(i, j, r)) + real(variable_tmp_1)*variable_tmp_2+imag(variable_tmp_1)*(eta+imag(exc_eigenvalues(q))));
					dielectric_tensor_bse(i, j, r).imag(imag(dielectric_tensor_bse(i, j, r)) + imag(variable_tmp_1)*variable_tmp_2-real(variable_tmp_1)*(eta+imag(exc_eigenvalues(q))));
				}
				dielectric_tensor_bse(i, j, r)=factor*dielectric_tensor_bse(i, j, r);
			}
	}
	for (int i = 0; i < 3; i++)
		for (int r = 0; r < number_of_energy_steps; r++)
			dielectric_tensor_bse(i, i, r).real(real(dielectric_tensor_bse(i, i, r)) + 1);
	cout <<"Calculation of the dielectric tensor finished" << endl;
	
	for (int r = 0; r < number_of_energy_steps; r++)
	{
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 3; j++)
				*file_diel << imag(dielectric_tensor_bse(i, j, r)) << " ";
		*file_diel << endl;
	}
};

int main()
{
	double fermi_energy = 1.6165;

	////Initializing Lattice
	Crystal_Lattice crystal;
	ifstream file_crystal_bravais;
	file_crystal_bravais.open("bravais.lattice.data");
	ifstream file_crystal_coordinates;
	file_crystal_coordinates.open("atoms.data");
	crystal.push_values(&file_crystal_bravais, &file_crystal_coordinates);
	file_crystal_bravais.close();
	file_crystal_coordinates.close();
	crystal.print();
	
	////Initializing k points list
	vec shift;
	shift.zeros(3);
	K_points k_points(&crystal,shift);
	ifstream file_k_points;
	file_k_points.open("k_points_list.dat");
	int number_k_points_list=14;
	k_points.push_k_points_list_values(&file_k_points,number_k_points_list);
	file_k_points.close();
	mat k_points_list=k_points.pull_k_points_list_values();
	
	//////Initializing g points list
	double cutoff_g_points_list=40;
	int dimension_g_points_list=2;
	vec direction_cutting(3);
	direction_cutting(0)=1; direction_cutting(1)=1; direction_cutting(2)=0;
	k_points.push_g_points_list_values(cutoff_g_points_list,dimension_g_points_list,direction_cutting);
	mat g_points_list=k_points.pull_g_points_list_values();
	int number_g_points_list=k_points.pull_number_g_points_list();
	//k_points.print();

	//////Initializing Coulomb potential
	double effective_dielectric_constant = 1.0;
	double minimum_k_point_modulus = 0.0001;
	int dimension_potential=2;
	vec direction_cutting_potential(3);
	direction_cutting_potential(0)=1; direction_cutting_potential(1)=1; direction_cutting_potential(2)=0;
	Coulomb_Potential coulomb_potential;
	coulomb_potential.push_values(&k_points,effective_dielectric_constant,minimum_k_point_modulus,dimension_potential,direction_cutting_potential);
	coulomb_potential.print();
	//int number_k_points_path=1000;
	//mat k_points_path; k_points_path.zeros(3,number_k_points_path);
	//for(int i=0;i<number_k_points_path;i++)
	//	k_points_path(0,i)=double(i);
	//coulomb_potential.print_profile(k_points_path,number_k_points_path);


	///In case you are interested in sampling the BZ instead
	///double spacing_fbz_k_points_list = 0.2;
	///k_points.push_k_points_list_values_fbz(&crystal, spacing_fbz_k_points_list, shift);
	///int number_k_points_list = k_points.pull_number_k_points_list();
	
	////Initializing the Tight Binding hamiltonian (saving the Wannier functions centers)
	ifstream file_htb;
	ifstream file_centers;
	string seedname;
	int number_atoms=crystal.pull_number_atoms();
	file_htb.open("tb_spin_polarized.dat");
	file_centers.open("tb_spin_polarized_centers.dat");
	Hamiltonian_TB htb;
	int spinorial_calculation = 1;
	/// 0 no spinors, 1 collinear spinors, 2 non-collinear spinors (implementing 0 and 1 cases)
	htb.push_values(&file_htb,&file_centers,fermi_energy,spinorial_calculation,number_atoms);
	int number_wannier_centers=htb.pull_number_wannier_functions();
	//htb.print_hamiltonian();

	//////Initializing dipole elements
	int number_conduction_bands_bse=8;
	int number_valence_bands_bse=8;
	Dipole_Elements dipole_elements;
	vec excitonic_momentum; excitonic_momentum.zeros(3);
	dipole_elements.push_values(number_k_points_list,k_points_list,number_g_points_list,g_points_list,number_wannier_centers,number_valence_bands_bse,number_conduction_bands_bse,&htb,spinorial_calculation);
	//dipole_elements.print(excitonic_momentum,3);

	////Initializing dielectric function
	Dielectric_Function dielectric_function;
	dielectric_function.push_values(&dipole_elements,number_k_points_list,number_g_points_list,g_points_list,number_valence_bands_bse,number_conduction_bands_bse,&coulomb_potential,spinorial_calculation);
	//cx_double omega; omega=0.0;
	//double eta=0.00001;
	//double PPA=27.00;
	//dielectric_function.print(excitonic_momentum,omega,eta,PPA,0);
	//double limitq_0=0.0001;
	//int number_omegas_points=10;
	//double omega_max=4.00;
	//cx_vec omegas(number_omegas_points);
	//for(int i=0;i<number_omegas_points;i++)
	//	omegas(i)=double(i/number_omegas_points)*omega_max;
	//cx_vec susceptability=dielectric_function.pull_susceptability(excitonic_momentum,omegas,number_omegas_points,eta,limitq_0);
	//for(int i=0;i<number_omegas_points;i++)
	//	cout<<omegas(i)<<" "<<susceptability(i)<<endl;

	//////Initializing BSE hamiltonian
	double eta=0.1; double epsilon=0.1;
	int adding_screening=1;
	Excitonic_Hamiltonian htbse;
	htbse.push_values(number_valence_bands_bse,number_conduction_bands_bse,&coulomb_potential,&dielectric_function,&htb,&dipole_elements,k_points_list,number_k_points_list,g_points_list,number_g_points_list,spinorial_calculation,adding_screening);
	htbse.print(excitonic_momentum,epsilon,eta);

	//////calculation dielectric matrix and optical spectrum
	//double scissor_operator=0.00;
	//ofstream file_diel;
	//file_diel.open("BSE.dat");
	//double energy_step=0.001;
	//double max_energy=4;
	//htbse.pull_dielectric_tensor_bse(excitonic_momentum,eta,epsilon,&file_diel,scissor_operator,energy_step,max_energy);
	////htbse.pull_excitonic_hamiltonian(excitonic_momentum, epsilon, eta);
	////htbse.print(excitonic_momentum,epsilon,eta);
	//file_diel.close();
	//file_htb.close();
	//
	return 1;
}
