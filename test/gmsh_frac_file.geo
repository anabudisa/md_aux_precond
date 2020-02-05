
// Define points
p0 = newp; Point(p0) = {0.0, 0.0, 0.0, 0.125 };
p1 = newp; Point(p1) = {1.0, 0.0, 0.0, 0.125 };
p2 = newp; Point(p2) = {1.0, 1.0, 0.0, 0.125 };
p3 = newp; Point(p3) = {0.0, 1.0, 0.0, 0.125 };
p4 = newp; Point(p4) = {0.0, 0.5, 0.0, 0.125 };
p5 = newp; Point(p5) = {1.0, 0.5, 0.0, 0.125 };
p6 = newp; Point(p6) = {0.5, 0.0, 0.0, 0.125 };
p7 = newp; Point(p7) = {0.5, 1.0, 0.0, 0.125 };
p8 = newp; Point(p8) = {0.5, 0.75, 0.0, 0.125 };
p9 = newp; Point(p9) = {1.0, 0.75, 0.0, 0.125 };
p10 = newp; Point(p10) = {0.75, 0.5, 0.0, 0.125 };
p11 = newp; Point(p11) = {0.75, 1.0, 0.0, 0.125 };
p12 = newp; Point(p12) = {0.5, 0.625, 0.0, 0.125 };
p13 = newp; Point(p13) = {0.75, 0.625, 0.0, 0.125 };
p14 = newp; Point(p14) = {0.625, 0.5, 0.0, 0.125 };
p15 = newp; Point(p15) = {0.625, 0.75, 0.0, 0.125 };
p16 = newp; Point(p16) = {0.5, 0.5, 0.0, 0.125 };
p17 = newp; Point(p17) = {0.75, 0.75, 0.0, 0.125 };
p18 = newp; Point(p18) = {0.625, 0.625, 0.0, 0.125 };
// End of point specification

// Start of specification of domain// Define lines that make up the domain boundary
bound_line_0 = newl;
Line(bound_line_0) ={p0, p6};
bound_line_1 = newl;
Line(bound_line_1) ={p6, p1};
Physical Line("DOMAIN_BOUNDARY_0") = { bound_line_0, bound_line_1 };
bound_line_2 = newl;
Line(bound_line_2) ={p1, p5};
bound_line_3 = newl;
Line(bound_line_3) ={p5, p9};
bound_line_4 = newl;
Line(bound_line_4) ={p9, p2};
Physical Line("DOMAIN_BOUNDARY_1") = { bound_line_2, bound_line_3, bound_line_4 };
bound_line_5 = newl;
Line(bound_line_5) ={p2, p11};
bound_line_6 = newl;
Line(bound_line_6) ={p11, p7};
bound_line_7 = newl;
Line(bound_line_7) ={p7, p3};
Physical Line("DOMAIN_BOUNDARY_2") = { bound_line_5, bound_line_6, bound_line_7 };
bound_line_8 = newl;
Line(bound_line_8) ={p3, p4};
bound_line_9 = newl;
Line(bound_line_9) ={p4, p0};
Physical Line("DOMAIN_BOUNDARY_3") = { bound_line_8, bound_line_9 };

// Line loop that makes the domain boundary
Domain_loop = newll;
Line Loop(Domain_loop) = {bound_line_0, bound_line_1, bound_line_2, bound_line_3, bound_line_4, bound_line_5, bound_line_6, bound_line_7, bound_line_8, bound_line_9};
domain_surf = news;
Plane Surface(domain_surf) = {Domain_loop};
Physical Surface("DOMAIN") = {domain_surf};
// End of domain specification

// Start specification of fractures/compartment boundary/auxiliary elements
frac_line_0 = newl; Line(frac_line_0) = {p4, p16};
Line{frac_line_0} In Surface{domain_surf};
frac_line_1 = newl; Line(frac_line_1) = {p5, p10};
Line{frac_line_1} In Surface{domain_surf};
frac_line_2 = newl; Line(frac_line_2) = {p10, p14};
Line{frac_line_2} In Surface{domain_surf};
frac_line_3 = newl; Line(frac_line_3) = {p14, p16};
Line{frac_line_3} In Surface{domain_surf};
Physical Line("FRACTURE_4") = { frac_line_0, frac_line_1, frac_line_2, frac_line_3 };

frac_line_4 = newl; Line(frac_line_4) = {p6, p16};
Line{frac_line_4} In Surface{domain_surf};
frac_line_5 = newl; Line(frac_line_5) = {p7, p8};
Line{frac_line_5} In Surface{domain_surf};
frac_line_6 = newl; Line(frac_line_6) = {p8, p12};
Line{frac_line_6} In Surface{domain_surf};
frac_line_7 = newl; Line(frac_line_7) = {p12, p16};
Line{frac_line_7} In Surface{domain_surf};
Physical Line("FRACTURE_5") = { frac_line_4, frac_line_5, frac_line_6, frac_line_7 };

frac_line_8 = newl; Line(frac_line_8) = {p8, p15};
Line{frac_line_8} In Surface{domain_surf};
frac_line_9 = newl; Line(frac_line_9) = {p9, p17};
Line{frac_line_9} In Surface{domain_surf};
frac_line_10 = newl; Line(frac_line_10) = {p15, p17};
Line{frac_line_10} In Surface{domain_surf};
Physical Line("FRACTURE_6") = { frac_line_8, frac_line_9, frac_line_10 };

frac_line_11 = newl; Line(frac_line_11) = {p10, p13};
Line{frac_line_11} In Surface{domain_surf};
frac_line_12 = newl; Line(frac_line_12) = {p11, p17};
Line{frac_line_12} In Surface{domain_surf};
frac_line_13 = newl; Line(frac_line_13) = {p13, p17};
Line{frac_line_13} In Surface{domain_surf};
Physical Line("FRACTURE_7") = { frac_line_11, frac_line_12, frac_line_13 };

frac_line_14 = newl; Line(frac_line_14) = {p12, p18};
Line{frac_line_14} In Surface{domain_surf};
frac_line_15 = newl; Line(frac_line_15) = {p13, p18};
Line{frac_line_15} In Surface{domain_surf};
Physical Line("FRACTURE_8") = { frac_line_14, frac_line_15 };

frac_line_16 = newl; Line(frac_line_16) = {p14, p18};
Line{frac_line_16} In Surface{domain_surf};
frac_line_17 = newl; Line(frac_line_17) = {p15, p18};
Line{frac_line_17} In Surface{domain_surf};
Physical Line("FRACTURE_9") = { frac_line_16, frac_line_17 };

// End of /compartment boundary/auxiliary elements specification

// Start physical point specification
Physical Point("FRACTURE_POINT_0") = {p8};
Physical Point("FRACTURE_POINT_1") = {p10};
Physical Point("FRACTURE_POINT_2") = {p12};
Physical Point("FRACTURE_POINT_3") = {p13};
Physical Point("FRACTURE_POINT_4") = {p14};
Physical Point("FRACTURE_POINT_5") = {p15};
Physical Point("FRACTURE_POINT_6") = {p16};
Physical Point("FRACTURE_POINT_7") = {p17};
Physical Point("FRACTURE_POINT_8") = {p18};
// End of physical point specification

