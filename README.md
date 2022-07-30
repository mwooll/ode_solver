# ode_solver
a project to solve 2nd order ordinary differential equations using a multigrid method

it can only solve equations of the form
  -(au')' = f in I = ]b, c[
  u = 0 on dI = {b, c}

This means that if we have a function g, which is twice differentiable in an intervall I and 0 on the borders. And we set f = -g" and a = 1, then the solution u will be a discretization of g.

This project arose as the main component of "practical training in numerics" a seminar I took at the University of Zurich given by Prof. Dr. S. A. Sauter.
During the realization of the project I got assisted by Francesco Florian a PhD student of Prof. Sauter.
