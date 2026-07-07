data{

    int<lower=1> T;

    int<lower=1> K;

    vector[T] y;

    matrix[T,K] X;

}

parameters{

    real mu;

    real<lower=0,upper=0.999> phi;

    real<lower=0> sigma_eta;

    vector[K] beta;

    vector[T] z;

    real<lower=2> nu;

}

transformed parameters{

    vector[T] h;

    h[1]=mu+z[1]*sigma_eta/sqrt(1-square(phi));

    for(t in 2:T){

        h[t]=
            mu
            +phi*(h[t-1]-mu)
            +dot_product(beta,X[t])
            +sigma_eta*z[t];

    }

}

model{

    mu~normal(-2,2);

    phi~beta(20,1.5);

    sigma_eta~normal(0,0.3);

    beta~normal(0,1);

    nu~gamma(2,0.1);

    z~normal(0,1);

    y~student_t(
        nu,
        0,
        exp(h/2)
    );

}