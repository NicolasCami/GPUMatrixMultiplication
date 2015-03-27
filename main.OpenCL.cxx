#include "qclcontext.h"
#include <ctime>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>

void initMatrix(int* m, int size) {
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            m[i*size+j] = rand()%(size*10);
        }
    }
}

void resetMatrix(int* m, int size) {
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            m[i*size+j] = 0;
        }
    }
}

void printMatrix(int* m, int size) {
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            std::cout << m[i*size+j] << "\t";
        }
        std::cout << "\n";
    }
}

void multMatrix(int* a, int* b, int* c, int size) {
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            int sum = 0;
            for(int k=0; k<size; k++) {
                sum += a[i*size+k] * b[k*size+j];
            }
            c[i*size+j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    int size = 4;
    bool gpu = false;
    QCLContext context;
    QCLProgram program;
    QCLKernel kernel;
    QCLVector<int> qcl_a;
    QCLVector<int> qcl_b;
    QCLVector<int> qcl_c;
    std::clock_t    start;
    int bloc_size = 1024;

    start = std::clock();

    if(argc > 1) {
        size = atoi(argv[1]);
    }
    if(argc > 2) {
        if(!strcmp(argv[2], "-gpu")) {
            gpu = true;
        }
    }

    int* a = new int[size*size];
    int* b = new int[size*size];
    int* c = new int[size*size];
    int* c_temp = new int[bloc_size*bloc_size];
    int* m_temp = new int[bloc_size*bloc_size];

    if(!context.create()) {
        std::cout << "can't create context\n";
        exit(1);
    }

    std::cout << "Matrix size : " << size << "\n"  << "Use GPU : " << ((gpu) ? "yes" : "no" ) << "\n";

    initMatrix(a, size);
    initMatrix(b, size);
    if(size < 5) {
        std::cout << "Matrix A :\n";
        printMatrix(a, size);
        std::cout << "Matrix B :\n";
        printMatrix(b, size);
    }

    // start timer
    start = std::clock();

    if(gpu) {

        program = context.buildProgramFromSourceFile("./Max.cl");
        kernel = program.createKernel("Multiply");

        if(size <= bloc_size) {

            // if matrix size is <= bloc_size
            // then we apply a simple matrix multiplication

            // GPU initialization
            qcl_a = context.createVector<int>(size*size, QCLMemoryObject::ReadOnly);
            qcl_b = context.createVector<int>(size*size, QCLMemoryObject::ReadOnly);
            qcl_c = context.createVector<int>(size*size, QCLMemoryObject::WriteOnly);
            kernel.setGlobalWorkSize(size, size);
            kernel.setArg(0, qcl_a);
            kernel.setArg(1, qcl_b);
            kernel.setArg(2, qcl_c);
            kernel.setArg(3, size);

            qcl_a.write(a, size*size);
            qcl_b.write(b, size*size);
            kernel.run();
            qcl_c.read(c, size*size);
        }
        else {

            // if matrix size is >= bloc_size
            // then we apply a bloc matrix multiplication

            int nbBlocs = size / bloc_size;
            if(nbBlocs*bloc_size < size) {
                nbBlocs++;
            }
            std::cout << "Multiply " << nbBlocs*nbBlocs << " matrix blocs..." << "\n";

            // GPU initialization
            qcl_a = context.createVector<int>(bloc_size*bloc_size, QCLMemoryObject::ReadOnly);
            qcl_b = context.createVector<int>(bloc_size*bloc_size, QCLMemoryObject::ReadOnly);
            qcl_c = context.createVector<int>(bloc_size*bloc_size, QCLMemoryObject::WriteOnly);
            kernel.setGlobalWorkSize(bloc_size, bloc_size);
            kernel.setArg(0, qcl_a);
            kernel.setArg(1, qcl_b);
            kernel.setArg(2, qcl_c);
            kernel.setArg(3, bloc_size);

            int no_bloc = 0;
            for(int i=0; i<nbBlocs; i++) {
                for(int j=0; j<nbBlocs; j++) {
                    resetMatrix(c_temp, bloc_size);

                    for(int k=0; k<nbBlocs; k++) {
                        // copy A on GPU
                        for(int ii=0; ii<bloc_size; ii++) {
                            for(int jj=0; jj<bloc_size; jj++) {
                                if(jj+k*bloc_size < size && ii < size)
                                    m_temp[ii*bloc_size + jj] = a[(ii*size) + (jj+k*bloc_size)];
                                else
                                    m_temp[ii*bloc_size + jj] = 0;
                            }
                        }
                        qcl_a.write(m_temp, bloc_size*bloc_size);

                        // copy B on GPU
                        for(int ii=0; ii<bloc_size; ii++) {
                            for(int jj=0; jj<bloc_size; jj++) {
                                if(ii+k*bloc_size < size && jj < size)
                                    m_temp[ii*bloc_size + jj] = b[((ii+k*bloc_size)*size) + jj];
                                else
                                    m_temp[ii*bloc_size + jj] = 0;
                            }
                        }
                        qcl_b.write(m_temp, bloc_size*bloc_size);

                        kernel.run();

                        // copy C from GPU to CPU
                        qcl_c.read(m_temp, bloc_size*bloc_size);
                        for(int ii=0; ii<bloc_size*bloc_size; ii++) {
                            c_temp[ii] += m_temp[ii];
                        }
                    }

                    // copy the calculated bloc
                    for(int ii=0; ii<bloc_size && ii+i*bloc_size<size; ii++) {
                        for(int jj=0; jj<bloc_size && jj+j*bloc_size<size; jj++) {
                            c[(ii+i*bloc_size)*size + (jj+j*bloc_size)] = c_temp[ii*bloc_size + jj];
                        }
                    }

                    no_bloc++;
                    std::cout << (no_bloc*100)/(nbBlocs*nbBlocs) << "%..." << "\n";
                }
            }
        }

        context.sync();

    }
    else {
        multMatrix(a, b, c, size);
    }

    if(size < 5) {
        std::cout << "Matrix C :\n";
        printMatrix(c, size);
    }

    std::cout << "Multiplication delay : " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms\n";

    // free memory
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] c_temp;
    delete[] m_temp;
}
