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
    </tr>
  </tbody>
</table>
