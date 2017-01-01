# LEEA

In this repository, I will experiment with the [limited-evaluation evolutionary algorithm](http://eplex.cs.ucf.edu/papers/morse_gecco16.pdf) (LEEA). Mostly, I want to see if I can get it to work on larger problems.

# Results

Here, I will put the best results as I get them:

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Pop.</th>
      <th>Mutation</th>
      <th>Cross-over</th>
      <th>Batch</th>
      <th># Generations</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MNIST</td>
      <td>512</td>
      <td>0.005</td>
      <td>0.5 * 0.999<sup>t</sup></td>
      <td>64</td>
      <td>1225</td>
      <td>81.59% success</td>
    </tr>
    <tr>
      <td>MNIST</td>
      <td>1024</td>
      <td>0.0005 + 0.05*0.995<sup>t</sup></td>
      <td>0.45 + 0.45*0.995<sup>t</sup></td>
      <td>64</td>
      <td>250</td>
      <td>78.54% success</td>
    </tr>
  </tbody>
</table>
