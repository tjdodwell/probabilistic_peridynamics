
elementSize = 2;

tabLength = 20;
totalWidth = 20;
innerWidth = 15;
taperWidth = totalWidth - innerWidth;
taperLength = 10;
innerLength = 115;

notchWidth = 5;
notchDepth = 5;


Point(1) = {0.0, 0.0, 0.0, elementSize};
Point(2) = {tabLength, 0.0, 0.0, elementSize};
Point(3) = {tabLength + taperLength, 0.5 * taperWidth, 0.0, elementSize};
Point(4) = {tabLength + taperLength + innerLength, 0.5 * taperWidth, 0.0, elementSize};
Point(5) = {tabLength + 2.0 * taperLength + innerLength, 0.0, 0.0, elementSize};
Point(6) = {2.0 * tabLength + 2.0 * taperLength + innerLength, 0.0, 0.0, elementSize};
Point(7) = {2.0 * tabLength + 2.0 * taperLength + innerLength, totalWidth, 0.0, elementSize};
Point(8) = {tabLength + 2.0 * taperLength + innerLength, totalWidth, 0.0, elementSize};
Point(9) = {tabLength + taperLength + innerLength, totalWidth - 0.5 * taperWidth, 0.0, elementSize};
Point(10) = {tabLength + taperLength + 0.5 * (innerLength + notchWidth), totalWidth - 0.5 * taperWidth, 0.0, elementSize};
Point(11) = {tabLength + taperLength + 0.5 * innerLength, totalWidth - 0.5 * taperWidth - notchDepth, 0.0, elementSize};
Point(12) = {tabLength + taperLength + 0.5 * (innerLength - notchWidth), totalWidth - 0.5 * taperWidth, 0.0, elementSize};
Point(13) = {tabLength + taperLength, totalWidth - 0.5 * taperWidth, 0.0, elementSize};
Point(14) = {tabLength, totalWidth, 0.0, elementSize};
Point(15) = {0.0, totalWidth, 0.0, elementSize};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,9};
Line(9) = {9,10};
Line(10) = {10,11};
Line(11) = {11,12};
Line(12) = {12,13};
Line(13) = {13,14};
Line(14) = {14,15};
Line(15) = {15,1};

Line Loop(6) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
Plane Surface(6) = {6};
