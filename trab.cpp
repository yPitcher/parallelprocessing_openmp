#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <math.h>
#include <cctype>
#include <algorithm>
#include <experimental/filesystem>
#include <cstdio>

#include "omp.h"
#include "csv.hpp"

#ifdef __linux__
    #include <unistd.h>
#endif
#ifdef _WIN32
    #include <windows.h>
    #pragma execution_character_set( "utf-8" )
#endif

static std::string RESULT_FOLDER = "result";
static std::string RESULT_EXTENSION = "txt";
static std::string DATASET = "./datasets/dataset.csv";
static std::string CURRENT_PATH = std::experimental::filesystem::current_path();

unsigned long long define_buffer_size(const size_t& columns_count) {
    /*
    * Calcula o buffer que ira ser utilizado para a leitura do arquivo
    */
    #ifdef _WIN32
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        unsigned long long total_memory = status.ullTotalPhys;
        unsigned long long available_memory = status.ullAvailPhys;

        // utilizaremos apenas 90% da memoria disponivel
        return static_cast<unsigned long long>(std::floor(((available_memory * 0.90) / columns_count) / 4));
    #endif

    #ifdef __linux__
        unsigned long long pages = sysconf(_SC_PHYS_PAGES);
        unsigned long long page_size = sysconf(_SC_PAGE_SIZE);
        unsigned long long available_pages = sysconf(_SC_AVPHYS_PAGES);
        unsigned long long total_memory = pages * page_size;

        // A quantidade de memoria via _SC_AVPHYS_PAGES pega apenas o total de memoria
        // que um programa pode usar sem interferir em outro, não é a memoria disponivel REAL!
        unsigned long long available_memory = available_pages * page_size;

        return static_cast<unsigned long long>(std::floor((available_memory / columns_count)));
    #endif
}

std::string gen_filepath(const std::string& filename) {
    return CURRENT_PATH + "/" + RESULT_FOLDER + "/" + filename + "." + RESULT_EXTENSION;
}

std::string gen_filepath(const std::string& filename, const std::string& extension) {
    return CURRENT_PATH + "/" + RESULT_FOLDER + "/" + filename + "." + extension;
}

void merge(const std::string& merged_file_name, const std::map<int, std::string>& dtypes, const std::vector<std::unordered_map<std::string, int>>& ids, const std::vector<std::string>& header) {
    /*
    * Junta os dados computados em arquivos separados de cada coluna e cria um arquivo final
    * com todos os dados categorizados com base nos id's criados nos arquivos separados.
    */
    std::ofstream combined_file;
    combined_file.open(gen_filepath(merged_file_name, "csv"), std::ios_base::app);
    csv::CSVReader reader(DATASET);
    size_t header_size = header.size();
    int idx = 0;

    // Insere os nomes das colunas no arquivo final
    for (const auto& val : reader.get_col_names()) {
        if (idx != header_size - 1) {
            combined_file << val << ",";
            idx++;
            continue;
        }
        idx++;
        combined_file << val << '\n';
    }

    idx = 0;

    for (const auto& row : reader) {
        for (auto& field : row) {
            if (dtypes.find(idx)->second == "numeric") {
                (idx != header_size - 1)
                    ? combined_file << field.get<csv::string_view>() << ","
                    : combined_file << field.get<csv::string_view>();
                idx++;
                continue;
            }
            (idx != header_size - 1)
                ? combined_file << ids[idx].find(field.get<>())->second << ","
                : combined_file << ids[idx].find(field.get<>())->second;
            idx++;
        }
        idx = 0;
        combined_file << std::endl;
    }

    combined_file.close();
}

void process_alpha(const std::vector<std::string>& vec, std::vector<std::unordered_map<std::string, int>>& ids, const std::vector<std::string>& header, const int column_id) {
    /*
    * Processa os dados do tipo 'alpha', escrevendo no arquivo apenas os dados unicos.
    */
    std::ofstream file;
    file.open(gen_filepath(header[column_id]), std::ios_base::app);
    int header_size = header.size();

    #pragma omp parallel for
    for (int j = column_id; j < vec.size(); j += header_size) {
        // Verifica se o valor já foi computado no map
        // se sim, o ignora e vai para a proxima iteração
        // ids[column_id].contains(vec[j]) <- C++20
        if (ids[column_id].find(vec[j]) != ids[column_id].end()) {
            continue;
        }
        // Caso ele não exista no map, o mesmo é inserido no map e escrito no arquivo.
        #pragma omp critical
        {
            ids[column_id].insert({ vec[j], ids[column_id].size() + 1 });
            file << ids[column_id].size() << '\n';
        }
    }
    file.close();
}

void process_numeric(const std::vector<std::string>& vec, const std::vector<std::string>& header, const int column_id) {
    /*
    * Processa os dados numericos, que neste caso é apenas escrever o conteudo da coluna em um arquivo
    * separado.
    */
    std::ofstream file;
    file.open(gen_filepath(header[column_id]), std::ios::app);
    int header_size = header.size();

    #pragma omp parallel for
    for (int j = column_id; j < vec.size(); j += header_size) {
        file << vec[j] << '\n';
    }
    file.close();
}

void create_process_tasks(const std::map<int, std::string>& dtypes, const std::vector<std::string>& vec, std::vector<std::unordered_map<std::string, int>>& ids, const std::vector<std::string>& header) {
    /*
    * Cria uma thread(omp task) para cada coluna com base no seu tipo
    */
    #pragma omp parallel
    {
        #pragma omp single
        for (const auto& pair : dtypes) {
            if (pair.second == "alpha") {
                #pragma omp task
                {
                    process_alpha(vec, ids, header, pair.first);
                }
                continue;
            }
            #pragma omp task
            process_numeric(vec, header, pair.first);
        }
    }
}

bool is_number(const std::string& str) {
    return !str.empty() && std::find_if(str.begin(), str.end(), [](unsigned char c) { return !std::isdigit(c); }) == str.end();
}

void get_dtypes(std::map<int, std::string>& dtypes, const std::vector<std::string>& vec, const std::vector<std::string>& header) {
    /*
    * Analisa os dados da coluna para saber o tipo de dados da coluna
    * e retorna um mapa com a chave sendo o id da coluna e o tipo dela, podendo ser
    * numeric ou alpha
    */
    int header_size = header.size();
    std::string current_type;
    for (int i = 0; i < header_size; i++) {
        for (int j = i; j < vec.size(); j += header_size) {
            if (is_number(vec[j])) {
                current_type = "numeric";
            }
            else {
                current_type = "alpha";
                break;
            }
        }
        dtypes.insert({ i, current_type });
    }
}

int main(int argc, char** argv) {
    #ifdef _WIN32
        SetConsoleOutputCP(CP_UTF8);
    #endif

    if (argc == 2) {
        DATASET = argv[1];
    }
    else if (argc == 1) {
        std::cout << "ERRO: informe um arquivo csv!" << '\n';
        std::cout << "Exemplo: ./programa ./caminho/dataset.csv" << std::endl;
        return 1;
    }
    else {
        DATASET = argv[1];
        std::cout << "AVISO: Argumentos a mais do que o esperado! Considerando apenas o primeiro!" << std::endl;
    }

    std::vector<std::string> vec;
    std::vector<std::string> header;
    std::vector<std::unordered_map<std::string, int>> ids;
    std::map<int, std::string> dtypes;

    vec.reserve(1'500'000);

    csv::CSVReader reader(DATASET);

    for (const auto& col : reader.get_col_names()) {
        header.emplace_back(col);
        ids.push_back(std::unordered_map<std::string, int>());
    }

    std::experimental::filesystem::path dataset_p{ DATASET };
    auto dataset_size = std::experimental::filesystem::file_size(dataset_p);

    unsigned long long CURRENT_BUFFER_SIZE = 0;
    assert(header.size() != 0);
    unsigned long long MAX_BUFFER_SIZE = define_buffer_size(header.size());

    int CHUNKS_PROCESSED = 0;
    int CHUNKS_TO_PROCESS = (((dataset_size / MAX_BUFFER_SIZE) * 2) != 0) ? ((dataset_size / MAX_BUFFER_SIZE) * 2) : 1;
    std::cout << "Lendo..." << std::endl;

    csv::CSVRow row;
    while (reader.read_row(row)) {
        if (((CURRENT_BUFFER_SIZE * 4) + row.size()) > MAX_BUFFER_SIZE) {

            // Considera o primeiro 'chunk' de de dados como o tipo da coluna para o resto do programa
            if (dtypes.size() == 0) {
                get_dtypes(dtypes, vec, header);
            }
            std::cout << "\r" << "PROCESSED: " << CHUNKS_PROCESSED << "~" << (dataset_size / MAX_BUFFER_SIZE) * 2 << std::flush;
            create_process_tasks(dtypes, vec, ids, header);
            CHUNKS_PROCESSED++;
            vec.clear();
            CURRENT_BUFFER_SIZE = 0;
        }
        CURRENT_BUFFER_SIZE += row.size() * 4;
        for (csv::CSVField& f : row) {
            vec.emplace_back(f.get<>());
        }
    }

    if (reader.eof() && CURRENT_BUFFER_SIZE != 0) {
        std::cout << "\r" << "PROCESSED: " << CHUNKS_PROCESSED << "~" << CHUNKS_TO_PROCESS << std::flush;
        if (dtypes.size() == 0) {
            get_dtypes(dtypes, vec, header);
        }
        create_process_tasks(dtypes, vec, ids, header);
        CHUNKS_PROCESSED++;
        std::cout << "\r" << "PROCESSED: " << CHUNKS_PROCESSED << "~" << CHUNKS_TO_PROCESS << std::flush;
    }

    std::cout << "\nJuntando arquivos" << std::endl;
    merge("merged", dtypes, ids, header);
    std::cout << "CSV gerado -> " << gen_filepath("merged", "csv") << std::endl;

    return 0;
}