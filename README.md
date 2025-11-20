### Предварительные сведения

Задача основана на численном расчете (интегрировании) дифференциальных уравнений распространения волн 
в неоднородной двумерной среде. Теоретическая часть и прототип алгоритма расчета рассмотрены на семинарах.


### Дано

Прямоугольная область с соотношением сторон 3:2 (разрешение не менее 900х600). 
Изображение области и размещенных в ней объектов находится в файле: `var_i.png`, где `i` - номер варианта.
В этой области размещены объекты (в зависимости от варианта): линзы, призмы, зеркала, экран. 
Источником света является всенаправленный точечный источник.

**Замечание**

1. Допускаются небольшие передвижения и вращения объектов, качественно не изменяющие конфигурацию объектов.
2. Ход луча изображен условно и может не совпадать с расчетом. Основной критерий корректности пути луча:
луч должен пройти через все призмы/линзы, отразиться от зеркал, после чего достигнуть экрана.  


**Граничные условия:**

- открытая граница (`open boundary`) - по краям области
- условия Дирихле - на зеркалах

**Константы задачи:**

- пространственный шаг решетки `h = 1.0` 
- скорость распространения волн `c = 1.0`
- временной шаг `dt = h / (c * 1.5)`
- константа численной схемы (для среды вне объектов) `kappa = c * dt / h`
- замедление красной, зеленой и синей волн внутри объектов:
  - `kappa / kappa_r = 1.30`
  - `kappa / kappa_g = 1.35`  
  - `kappa / kappa_b = 1.40`  


Требуется
Рассчитать амплитуду колебаний в двух точках комнаты Пенроуза: первая расположена в освещенной области и
максимально удалена от источника света, вторая - в неосвещенной области.
Подробности здесь: https://en.wikipedia.org/wiki/Illumination_problem

### Требуется

Написать алгоритм расчета распространения волн с использованием библиотеки `Taichi` <0>, а также:

  - выполнить симуляцию для трех случаев, изображенных по ссылке к задаче <1> (за каждый не реализованный случай)
  - подобрать необходимое количество итераций для получения возмущения в первой точке и увеличить их в 2 раза <1>
  - построить графики амплитуды от времени для каждой из двух точек <1> 
  - записать анимацию распространения волны <2>
  - записать анимацию накопительного кадра <2>

### Материалы

- [Wave equation](https://en.wikipedia.org/wiki/Wave_equation)
- [The 2D wave equation - Solution with the method of finite differences in Python](https://beltoforion.de/en/recreational_mathematics/2d-wave-equation.php)
- [Wave on a string (interactive demo)](https://phet.colorado.edu/en/simulations/wave-on-a-string/teaching-resources)
- [Finite difference methods for 2D and 3D wave equations](https://hplgit.github.io/fdm-book/doc/pub/book/sphinx/._book008.html)
- [Absorbing Boundary Conditions for the Finite-Difference Approximation of the Time-Domain Electromagnetic-Field Equations GERRIT MUR](http://home.cc.umanitoba.ca/~lovetrij/cECE4390/Notes/Mur,%20G.%20-%20Absorbing%20BCs%20for%20the%20Finite-Difference%20Approximation%20of%20the%20TD%20EM%20Eqs.%20-%201981pdf.pdf)
- [Дисперсия света](https://ru.wikipedia.org/wiki/Дисперсия_света)
- [Показатель преломления](https://ru.wikipedia.org/wiki/Показатель_преломления)
- [Impulse function | Graphtoy](https://graphtoy.com/?f1(x,t)=exp(-(x%5E2)/2.0/2%5E2)/2&v1=true&f2(x,t)=&v2=false&f3(x,t)=&v3=false&f4(x,t)=cos(20*x)*f1(x,t)&v4=true&f5(x,t)=&v5=false&f6(x,t)=&v6=false&grid=1&coords=0,0,4.205926793776742)
- [Light-Simulation-WebGL](https://github.com/ArtemOnigiri/Light-Simulation-WebGL)
- [Симуляция волн света на клеточных автоматах](https://www.youtube.com/watch?v=noUpBKY2rIg)
- [Finite difference methods for wave equations](https://hplgit.github.io/fdm-book/doc/pub/wave/pdf/wave-4print.pdf)
