# LEEA

In this repository, I will experiment with the [limited-evaluation evolutionary algorithm](http://eplex.cs.ucf.edu/papers/morse_gecco16.pdf) (LEEA). Mostly, I want to see if I can get it to work on larger problems.

# Results

Here, I will put the best results as I get them. The batch size for every experiment is 64. The models are:

 * FC: fully-connected network with standard LEEA. No weight decay is used. Mutations are simply Gaussian noise.
 * FC-S: fully-connected network with assignment-based mutation. Mutation overwrites certain weights with values sampled from a Gaussian.
 * FC-D: fully-connected network with weight-decay used to prevent weight explosion.

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Accuracy</th>
      <th>Model</th>
      <th>Pop.</th>
      <th>Mutation</th>
      <th>Cross-over</th>
      <th>Generations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MNIST</td>
      <td>90.51%</td>
      <td>FC-S</td>
      <td>512</td>
      <td>0.0001 + 0.01*0.996<sup>t</sup></td>
      <td>0.5</td>
      <td>3600</td>
    </tr>
    <tr>
      <td>MNIST</td>
      <td>84.97%</td>
      <td>FC</td>
      <td>1024</td>
      <td>0.05</td>
      <td>0.5</td>
      <td>2500</td>
    </tr>
    <tr>
      <td>MNIST</td>
      <td>81.59%</td>
      <td>FC</td>
      <td>512</td>
      <td>0.005</td>
      <td>0.5 * 0.999<sup>t</sup></td>
      <td>1225</td>
    </tr>
    <tr>
      <td>MNIST</td>
      <td>80.79%</td>
      <td>FC-D</td>
      <td>768</td>
      <td>0.0005 + 0.01*0.995<sup>t</sup></td>
      <td>0.5</td>
      <td>1967</td>
    </tr>
    <tr>
      <td>MNIST</td>
      <td>78.54%</td>
      <td>FC</td>
      <td>1024</td>
      <td>0.0005 + 0.05*0.995<sup>t</sup></td>
      <td>0.05 + 0.45*0.995<sup>t</sup></td>
      <td>250</td>
    </tr>
  </tbody>
</table>
