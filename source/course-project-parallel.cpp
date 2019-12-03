#include <hpx/hpx_main.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_algorithm.hpp>


#include <iostream>
#include <iomanip>
#include <cmath>
#include <utility>
#include <list>
#include <vector>
#include <fstream>
#include <exception>
#include <algorithm>
static const double pi = 3.1415926535897;


struct vec_3
{
	double x, y, z;

	//constructors
	vec_3() noexcept : x(0.0), y(0.0), z(0.0) {}
	vec_3(double x, double y, double z) noexcept : x(x), y(y), z(z) {}
	vec_3(const vec_3 & other) noexcept : x(other.x), y(other.y), z(other.z) {}
	vec_3(vec_3 && other) noexcept = default;

	double & operator [] (size_t idx)
	{
		if (2 < idx)
		{
			throw std::out_of_range("Accessing element not in vec_3");
		}
		return idx == 0 ? x : (idx == 1 ? y : z);
	}

	const double & operator [](size_t idx) const
	{
		if (2 < idx)
		{
			throw std::out_of_range("Accessing element not in vec_3");
		}

		return idx == 0 ? x : (idx == 1 ? y : z);
	}

	// operator overloads
	vec_3 & operator = (const vec_3 & other) noexcept = default;

	vec_3 & operator = (vec_3 && other) noexcept = default;

	vec_3 & operator += (const vec_3 & rhs) noexcept
	{
		x += rhs.x;
		y += rhs.y;
		z += rhs.z;
		return *this;
	}

	vec_3 & operator -= (const vec_3 & rhs) noexcept
	{
		x -= rhs.x;
		y -= rhs.y;
		z -= rhs.z;
		return *this;
	}

	vec_3 & operator *=(const vec_3 & rhs) noexcept
	{
		x = y * rhs.z - z * rhs.y;
		y = z * rhs.x - x * rhs.z;
		z = x * rhs.y - y * rhs.y;
		return *this;
	}

	vec_3 &  operator *=(double rhs) noexcept
	{
		x *= rhs;
		y *= rhs;
		z *= rhs;
		return *this;
	}

	friend vec_3 operator + (vec_3 lhs, const vec_3 & rhs) noexcept;
	friend vec_3 operator - (vec_3 lhs, const vec_3 & rhs) noexcept;
	friend vec_3 operator * (vec_3 lhs, const vec_3 & rhs) noexcept;
	friend vec_3 operator * (vec_3 lhs, const double rhs) noexcept;
	friend vec_3 operator * (const double lhs, vec_3  rhs) noexcept;
	friend vec_3 operator / (vec_3  lhs, const double rhs) noexcept;
	friend vec_3 operator / (const  double lhs, vec_3  rhs) noexcept;

	friend bool operator == (const vec_3& lhs, const vec_3 & rhs) noexcept;
	friend bool operator != (const vec_3& lhs, const vec_3 & rhs) noexcept;
	friend std::ostream & operator << (std::ostream & out, const vec_3 & v) noexcept;
	friend std::istream & operator >> (std::istream & in, const vec_3 &  v) noexcept;

	//other mathematic functions
	double norm() const noexcept
	{
		return std::sqrt(x*x + y * y + z * z);
	}

	friend double dot(const vec_3 & lhs, const vec_3 & rhs) noexcept;
	friend double distance(const vec_3 & lhs, const vec_3 & rhs) noexcept;
};

vec_3 operator + (vec_3 lhs, const vec_3 & rhs) noexcept
{
	return lhs += rhs;
}

vec_3 operator - (vec_3 lhs, const vec_3 & rhs) noexcept
{
	return lhs -= rhs;
}

vec_3 operator * (vec_3 lhs, const vec_3 & rhs) noexcept
{
	return lhs *= rhs;
}

vec_3 operator * (vec_3 lhs, const double rhs) noexcept
{
	return lhs *= rhs;
}
vec_3 operator * (const double lhs, vec_3 rhs) noexcept
{
	return rhs *= lhs;
}
vec_3 operator / (vec_3 lhs, const double rhs) noexcept
{
	return lhs *= 1 / rhs;
}

vec_3 operator / (const double lhs, vec_3 rhs) noexcept
{
	return rhs *= lhs;
}

double dot(const vec_3 & lhs, const vec_3 & rhs) noexcept
{
	return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

double distance(const vec_3 & lhs, const vec_3 & rhs) noexcept
{
	return (lhs - rhs).norm();
}

bool operator == (const vec_3& lhs, const vec_3 & rhs) noexcept
{
	return !(std::isless(lhs.x, rhs.x) || !std::isless(rhs.x, lhs.x)) &&
		!(std::isless(lhs.y, rhs.y) || !std::isless(rhs.y, lhs.y)) &&
		!(std::isless(lhs.z, rhs.z) || !std::isless(rhs.z, lhs.z));
}

bool operator != (const vec_3& lhs, const vec_3 & rhs) noexcept
{
	return !(lhs == rhs);
}

std::ostream & operator << (std::ostream & out, vec_3 & v) noexcept
{
	auto precision = out.precision();
	auto width = out.width();
	out << "(";
	out << std::setw(width) << std::setprecision(precision) << v.x << ",";
	out << std::setw(width) << std::setprecision(precision) << v.y << ",";
	out << std::setw(width) << std::setprecision(precision) << v.z;
	out << ")";
	return out;
}

std::istream & operator >> (std::istream & in, vec_3 & v) noexcept
{
	return in >> v.x >> v.y >> v.z;
}

class peridynamic_solver
{
public:
	peridynamic_solver(double lengthX, double lengthY, double lengthZ, double meshSize, std::string in_file, std::string out_file ) :
		lx(lengthX), ly(lengthY), lz(lengthZ), h(meshSize),
		mx(std::floor(lengthX / meshSize)), my(std::floor(lengthY / meshSize)), mz(std::floor(lengthZ / meshSize)),
		in_file_name(in_file), out_file_name(out_file)
	{
		initialize();
	}
	
	//just for testing
	peridynamic_solver(std::string infile_name, std::string outfile_name): in_file_name(infile_name), out_file_name(outfile_name)
	{
		
		lx = 10.0;
		ly = 10.0;
		lz = 0.0;
		h = 1.0;
		mx = std::floor(lx / h);
		my = std::floor(ly / h);
		mz = std::floor(lz / h);
		initialize();
		double f = 10;
		for (size_t i = 0; i < num_nodes; ++i)
		{
			if (positions[i].x <= horizon * h)
			{
				external_forces[i].x = f;
			}
		}
		for (size_t i = 0; i < num_nodes; ++i)
		{
			if ((mx - horizon) * h <= positions[i].x)
			{
				external_forces[i].x = -f;
			}
		}
	}
	void DoWork()
	{
		std::ofstream out_stream(out_file_name);
		out_stream.close();
		size_t num_ts = std::floor(final_time / time_step);
		for (curr_ts = 0; curr_ts < num_ts; ++curr_ts)
		{
			std::cout << curr_ts << " / " << num_ts << std::endl;
			update_positions();
			remove_damaged();
			compute_forces();
			compute_acceleration();
			compute_displacements();
			write_to_file();
		}
	}

private:
	double lx, ly, lz;
	size_t mx, my, mz;
	double h;
	double c, sc;
	size_t curr_ts;
	std::vector<double> mass_density;
	std::vector<vec_3> forces;
	std::vector <double> volumes;
	std::vector <vec_3> positions;
	std::vector<std::vector<std::pair<size_t, double>>> neighbors;
	std::vector<vec_3> external_forces;
	std::vector<vec_3> accelerations;
	std::vector<vec_3> displacements_curr;
	std::vector<vec_3> displacements_prev;
	std::vector<vec_3> displacements_next;
	//elements to remove from neighborhood		
	std::vector<std::pair<size_t, size_t>> neighbors_to_remove;
	double integration_constant = 1.0;
	//from file
	double k, k_ic, q;
	double final_time, time_step;
	hpx::lcos::local::spinlock remove_spinlock;
	double horizon;
	size_t num_nodes;
	std::string in_file_name;
	std::string out_file_name = "out.txt";

	void generate_volumes()
	{
		volumes.resize(num_nodes, (mx > 0 ? h : 1.0)  * (my > 0 ? h : 1.0)  * (mz > 0 ? h : 1));
	}
	void compute_forces()
	{
	hpx::parallel::for_loop(hpx::parallel::execution::par, 0,  num_nodes, 
		[&](const auto i)
		{
			
			forces[i] = vec_3();
			for (size_t j = 0; j < neighbors[i].size(); ++j)
			{
				size_t neighbor = neighbors[i][j].first;
				double init_distance = neighbors[i][j].second;
				double stretch_ij = compute_stretch(i, neighbor, init_distance);
				forces[i] += pairwise_force(i, neighbor, stretch_ij)  * volumes[neighbor];
				if (damage(stretch_ij))
				{
					//spinlock lock unlock here to avoid race-conditions with push_back
					remove_spinlock.lock();	
					neighbors_to_remove.push_back(std::make_pair(i, neighbor));
					remove_spinlock.unlock();
				}
			}
			forces[i] += external_forces[i];
		});
	}
	void compute_acceleration()
	{
		for (size_t i = 0; i < num_nodes; ++i)
		{
			accelerations[i] = forces[i] / mass_density[i];
		}
	}
	void compute_displacements()
	{
		for (size_t i = 0; i < num_nodes; ++i)
		{
			displacements_next[i] = 2 * displacements_curr[i] - displacements_prev[i] + time_step * time_step * (forces[i]);
		}
	}
	void update_positions()
	{
		std::swap(displacements_curr, displacements_prev);
		std::swap(displacements_next, displacements_curr);
		for (size_t i = 0; i < num_nodes; ++i)
		{
			positions[i] += displacements_curr[i];
		}
	}
	void read_file()
	{
		std::ifstream in_stream(in_file_name);
		if (!in_stream.is_open())
		{
			throw std::runtime_error("Could not open in file");
		}
		in_stream >> k	>> k_ic >> q >> final_time >> time_step >> horizon;
		in_stream.close();
	}
	void write_to_file()
	{
		std::ofstream out_stream(out_file_name, std::ostream::app);
		if (out_stream.is_open())
		{
			out_stream << "=== time " << time_step * curr_ts << " ===" << std::endl;
			out_stream << *this;
			out_stream.close();
		}
		else
		{
			throw std::runtime_error("Could not open file");
		}
	}

	void initialize()
	{
		num_nodes = (0 < mx ? mx : 1) * (0 < my ? my : 1) * (0 < mz ? mz : 1);
		read_file();
		c = compute_c();
		sc = compute_sc();
		displacements_prev.resize(num_nodes, vec_3());
		displacements_curr.resize(num_nodes, vec_3());
		displacements_next.resize(num_nodes, vec_3());
		neighbors.resize(num_nodes, std::vector<std::pair<size_t, double>>());
		
		forces.resize(num_nodes, vec_3());
		external_forces.resize(num_nodes, vec_3());
		
		mass_density.resize(num_nodes, 1.0);
		accelerations.resize(num_nodes, vec_3());
		generate_volumes();
		generate_mesh();
		find_neighbors();
		
	
	}

	void generate_mesh()
	{
		positions.reserve(num_nodes);
		//using 3D mesh
		if (0 < mx && 0 < my && 0 < mz)
		{
			for (size_t i = 0; i < mx; ++i)
			{
				for (size_t j = 0; j < my; ++j)
				{
					for (size_t k = 0; k < mz; ++k)
					{
						auto v = vec_3(h * i, h * j, h * k);
						positions.emplace_back(vec_3(h * i, h * j, h * k));
					}
				}
			}
		}
		//using 2d mesh
		else if (((mx != 0) + (my != 0) + (mz != 0)) == 2)
		{
			size_t num_values[3] = { mx, my, mz };
			size_t first_d = 0 < mx ? 0 : 1;
			size_t second_d = 0 < mz ? 2 : 1;
			for (size_t i = 0; i < num_values[first_d]; ++i)
			{
				for (size_t j = 0; j < num_values[second_d]; ++j)
				{
					vec_3 v;
					v[first_d] = i * h;
					v[second_d] = j * h;
					positions.push_back(std::move(v));
				}
			}
		}
		//only one dimension
		else
		{
			size_t num_values[3] = { mx, my, mz };
			size_t dimension = 0 < mx ? 0 : (0 < my ? 1 : 2);
			for (size_t i = 0; i < num_values[dimension]; ++i)
			{
				vec_3 v;
				v[dimension] = i * h;
				positions.push_back(std::move(v));
			}
		}
	}

	void find_neighbors()
	{
		neighbors.resize(num_nodes);
		hpx::parallel::for_loop (hpx::parallel::execution::par, 0,  num_nodes, [&] (const auto i)
		{
			for (int j = 0; j < num_nodes; ++j)
			{
				double dis = distance(positions[i], positions[j]);
				if (i != j && dis < horizon)
				{
					neighbors[i].push_back(std::make_pair(j, dis));
				}
			}
		});
	}

	void remove_damaged()
	{
		for (size_t i = 0; i < neighbors_to_remove.size(); ++i)
		{
			size_t neighbor_to_remove = neighbors_to_remove[i].second;
			neighbors[neighbors_to_remove[i].first].erase(std::remove_if(
				neighbors[neighbors_to_remove[i].first].begin(),
				neighbors[neighbors_to_remove[i].first].end(),
				[neighbor_to_remove](std::pair<size_t, float> neighbor)
			{
				return neighbor.first == neighbor_to_remove;
			}));
		}
		neighbors_to_remove.clear();
	}

	double compute_c()
	{
		return (18.0 * k) / (pi * horizon);
	}

	double compute_sc()
	{
		return (5.0 / 12.0) / std::sqrt(k_ic / (k * k * horizon));
	}

	double compute_stretch(const size_t i, const  size_t j, const double initial_distance)
	{
		return (distance(positions[j], positions[i]) - initial_distance) / initial_distance;
	}

	vec_3 pairwise_force(const size_t i, const size_t j, const double stretch)
	{
		return stretch * c * (positions[j] - positions[i]) / distance(positions[j], positions[i]);
	}

	bool damage(const double stretch)
	{
		return  sc < stretch;
	}
	friend std::ostream & operator << (std::ostream & out, peridynamic_solver & pd) noexcept;
	void print_grid(std::vector<vec_3> & v) 
	{
		for (int i = 0; i < num_nodes; ++i)
		{
			if (i % mx == 0)
				std::cout << std::endl;
			std::cout.precision(2);
			std::cout.width(7);
			std::cout << v[i] << " ";
		}
	}
};

std::ostream & operator << (std::ostream & out, peridynamic_solver & pd) noexcept
{
	for (size_t i = 0; i < pd.num_nodes; ++i)
	{
		out.precision(3);
		out.width(6);

		out << i << "|" << pd.positions[i] << "|" << pd.forces[i] << "|" << pd.accelerations[i] << "|" <<  pd.curr_ts << std::endl;
	}
	return out;
}

int main()
{
    peridynamic_solver pd {"../input_files/config.dat", "../output_files/output_paralllel.dat"};
    pd.DoWork();
    return EXIT_SUCCESS;
}

