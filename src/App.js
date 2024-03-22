import React from 'react'
import Plot from 'react-plotly.js'

export default class App extends React.Component {

  constructor(){
    super();

    // Params = epochs, window, output, learning_rate
    // lookback_days, prop

    this.state = {response: null,
                  epochs: 1000,
                  window: 100,
                  output: 30,
                  lr: 0.0001,
                  lookback: 50,
                  prop: 0.7,
                  ticker: 'AAPL',
                  sock: null,
                  stats: 'idle'}

    this.changeParam = this.changeParam.bind(this)
    this.clickData = this.clickData.bind(this)
    this.plotData = this.plotData.bind(this)
  }

  componentDidMount(){
    const socket = new WebSocket('ws://localhost:8080')
    socket.onmessage = (evt) => {
      // type, payload
      const response = JSON.parse(evt.data)
      if(response['type'] === 'update'){
        this.setState({ stats: response['payload']})
      } else {
        this.setState({ response: response['payload']})
      }
    }
    this.setState({ sock: socket})
  }

  changeParam(evt){
    this.setState({ [evt.target.name]: evt.target.value })
  }

  clickData(evt){
    const { sock, ticker, epochs, window, output, lr, lookback, prop } = this.state
    const msg = [ticker, epochs, window, output, lr, lookback, prop]
    sock.send(JSON.stringify(msg))
  }

  plotData(){
    const hold = []
    const { response } = this.state
    if(response !== null){
      hold.push(
        <Plot
          data={[{
            x: response[0],
            y: response[2],
            type: 'lines+markers',
            mode: 'lines',
            marker: {
              color: 'red'
            },
            name: 'Historical'
          },
          {
            x: response[1],
            y: response[3],
            type: 'lines+markers',
            mode: 'lines',
            marker: {
              color: 'limegreen'
            },
            name: 'Predicted'
          }]}
          layout={{
            title: 'Forecasted Stock'
          }}
        />
      )
    }
    return hold 
  }

  render(){

    const { epochs, window, output, lr, lookback, prop } = this.state

    return(
      <React.Fragment>
        <center>
          <div style={{backgroundColor: 'black', color: 'limegreen', fontSize: 40}}>Stock Price Forecaster</div>
          <br/>
          <br/>
          <center>
            <tr>
              <td>Epochs</td>
              <td>Window</td>
              <td>Output</td>
              <td>LearningRate</td>
              <td>Lookback</td>
              <td>Proportion</td>
            </tr>
            <tr>
              <td><input name="epochs" style={{backgroundColor: 'black', color: 'limegreen', width: 100, textAlign: 'center', fontSize: 20}} onChange={this.changeParam}/></td>
              <td><input name="window" style={{backgroundColor: 'black', color: 'limegreen', width: 100, textAlign: 'center', fontSize: 20}} onChange={this.changeParam}/></td>
              <td><input name="output" style={{backgroundColor: 'black', color: 'limegreen', width: 100, textAlign: 'center', fontSize: 20}} onChange={this.changeParam}/></td>
              <td><input name="lr" style={{backgroundColor: 'black', color: 'limegreen', width: 100, textAlign: 'center', fontSize: 20}} onChange={this.changeParam}/></td>
              <td><input name="lookback" style={{backgroundColor: 'black', color: 'limegreen', width: 100, textAlign: 'center', fontSize: 20}} onChange={this.changeParam}/></td>
              <td><input name="prop" style={{backgroundColor: 'black', color: 'limegreen', width: 100, textAlign: 'center', fontSize: 20}} onChange={this.changeParam}/></td>
            </tr>
            <tr>
              <td style={{fontSize: 25}}>{epochs}</td>
              <td style={{fontSize: 25}}>{window}</td>
              <td style={{fontSize: 25}}>{output}</td>
              <td style={{fontSize: 25}}>{lr}</td>
              <td style={{fontSize: 25}}>{lookback}</td>
              <td style={{fontSize: 25}}>{prop}</td>
            </tr>
          </center>
          <br/>
          <div>Enter Ticker</div>
          <input name="ticker" value={this.state.ticker} style={{backgroundColor: 'black', color: 'limegreen', width: 100, textAlign: 'center', fontSize: 20}} onChange={this.changeParam}/>
          <br/>
          <br/>
          <div><button style={{backgroundColor: 'black', color:  'limegreen', fontSize: 16}} onClick={this.clickData}>Fetch Data</button></div>
          <br/>
          <div>{this.state.stats}</div>
          <br/>
          <br/>
          <div>{this.plotData()}</div>
       </center>
      </React.Fragment>
    );
  }

}