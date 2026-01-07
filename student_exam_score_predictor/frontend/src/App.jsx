import { useState } from 'react'

function App() {
  const [hours, setHours] = useState('')
  const [sleep, setSleep] = useState('')
  const [attendance, setAttendance] = useState('')
  const [score, setScore] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setResult(null)

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          hours_studied: parseFloat(hours),
          sleep_hours: parseFloat(sleep),
          attendance_percent: parseFloat(attendance),
          previous_score: parseFloat(score),
        }),
      })

      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error('Error:', error)
      setResult({ error: 'Failed to fetch prediction' })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4 font-sans">
      <div className="bg-white p-8 rounded-xl shadow-2xl w-full max-w-md border border-gray-100">
        <h1 className="text-3xl font-extrabold text-blue-900 mb-6 text-center">
          Student Success <span className="text-blue-500">Predictor</span>
        </h1>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-gray-700 text-sm font-semibold mb-1" htmlFor="hours">
              Hours Studied (0-24)
            </label>
            <input
              id="hours"
              type="number"
              step="0.1"
              min="0"
              max="24"
              required
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition duration-200"
              placeholder="e.g. 5.5"
              value={hours}
              onChange={(e) => setHours(e.target.value)}
            />
          </div>

          <div>
            <label className="block text-gray-700 text-sm font-semibold mb-1" htmlFor="sleep">
              Sleep Hours (0-24)
            </label>
            <input
              id="sleep"
              type="number"
              step="0.1"
              min="0"
              max="24"
              required
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition duration-200"
              placeholder="e.g. 7.5"
              value={sleep}
              onChange={(e) => setSleep(e.target.value)}
            />
          </div>

          <div>
            <label className="block text-gray-700 text-sm font-semibold mb-1" htmlFor="attendance">
              Attendance % (0-100)
            </label>
            <input
              id="attendance"
              type="number"
              step="0.1"
              min="0"
              max="100"
              required
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition duration-200"
              placeholder="e.g. 85"
              value={attendance}
              onChange={(e) => setAttendance(e.target.value)}
            />
          </div>

          <div>
            <label className="block text-gray-700 text-sm font-semibold mb-1" htmlFor="score">
              Previous Score (0-100)
            </label>
            <input
              id="score"
              type="number"
              step="1"
              min="0"
              max="100"
              required
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none transition duration-200"
              placeholder="e.g. 85"
              value={score}
              onChange={(e) => setScore(e.target.value)}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className={`w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-lg transition duration-200 shadow-md ${
              loading ? 'opacity-70 cursor-not-allowed' : ''
            }`}
          >
            {loading ? 'Predicting...' : 'Predict Result'}
          </button>
        </form>

        {result && (
          <div className={`mt-8 p-6 rounded-lg text-center transform transition-all duration-300 bg-blue-50 text-blue-900 border border-blue-200`}>
            {result.error ? (
               <p className="font-semibold text-red-600">{result.error}</p>
            ) : (
              <>
                <p className="text-sm uppercase tracking-wide font-semibold text-gray-500 mb-1">Prediction</p>
                <p className="text-4xl font-black mb-2">{result.predicted_score}</p>
                <p className="text-sm font-medium opacity-80">
                  Expected Exam Score
                </p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
